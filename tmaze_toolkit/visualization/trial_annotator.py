"""
Trial Annotator GUI
===================
Interactive video tool for manually correcting missed trial detections.

For each row in ``trials_df`` where ``start_frame`` is NaN, the GUI
seeks to the approximate location in the video and lets the user mark the
trial start and end frame by pressing S / E (or clicking the buttons).

Keyboard shortcuts
------------------
  Space          Play / Pause
  S              Mark current frame as trial START
  E              Mark current frame as trial END
  N              Save annotation and go to NEXT missed trial
  P              Go back to PREVIOUS missed trial (clears its saved annotation)
  X              Skip current trial (leave as NaN)
  ← / →          Step 1 frame back / forward
  Shift + ← / →  Step 10 frames back / forward

Usage
-----
    from tmaze_toolkit.visualization.trial_annotator import annotate_trials

    updated_df = annotate_trials(
        video_path='session.mp4',
        trials_df=trials_df,
        hint_frame_col='floor_start_frame',   # optional column for seek hints
    )
"""

import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
import pandas as pd

try:
    from PIL import Image, ImageTk
except ImportError:
    raise ImportError(
        "Pillow is required for the trial annotator. "
        "Install it with:  pip install Pillow"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPEED_OPTIONS = [0.25, 0.5, 1.0, 2.0, 4.0]

_BG      = '#2b2b2b'
_BG2     = '#1e1e1e'
_BG3     = '#383838'
_BTN     = '#484848'
_FG      = '#dddddd'
_GREEN   = '#52b788'
_RED     = '#e07070'
_YELLOW  = '#f4a261'
_BLUE    = '#5b8dd9'


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------

class TrialAnnotatorApp:
    """Tkinter application for manually annotating missed trial boundaries."""

    def __init__(self, root: tk.Tk, video_path: str,
                 trials_df: pd.DataFrame, hint_frame_col: str = None):
        self.root = root
        self.root.title("Trial Annotator")
        self.root.configure(bg=_BG)

        self.trials_df = trials_df.copy()
        self.hint_frame_col = hint_frame_col

        # ── Video ────────────────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.fps          = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_w             = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h             = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Scale video to fit display (~900 px wide)
        self.disp_w = 900
        self.disp_h = int(vid_h * self.disp_w / vid_w)

        # ── Missed trials ────────────────────────────────────────────────────
        self.missed_indices = self.trials_df.index[
            self.trials_df['valid'] != 'yes'
        ].tolist()
        self.missed_pos = 0          # current position in missed_indices list

        # ── Playback state ───────────────────────────────────────────────────
        self.current_frame   = 0
        self.is_playing      = False
        self.speed           = 1.0
        self._after_id       = None
        self._photo          = None        # keep reference to prevent GC
        self._slider_dragging = False

        # ── Pending annotation for the current missed trial ──────────────────
        self.pending_start: int = None
        self.pending_end:   int = None

        self._build_ui()
        self._bind_keys()

        if not self.missed_indices:
            messagebox.showinfo(
                "No Missed Trials",
                "No rows with valid == 'no' found. Nothing to annotate."
            )
        else:
            self._load_current_trial()

        self._start_loop()

    # =========================================================================
    # UI construction
    # =========================================================================

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ── Video canvas ─────────────────────────────────────────────────────
        self.canvas = tk.Canvas(
            self.root,
            width=self.disp_w, height=self.disp_h,
            bg='black', highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, sticky='nsew', padx=6, pady=(6, 2))

        # ── Status bar ───────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Loading…")
        tk.Label(
            self.root, textvariable=self.status_var,
            bg=_BG2, fg='#888888', font=('Courier', 9),
            anchor='w', padx=10, pady=3,
        ).grid(row=1, column=0, sticky='ew')

        # ── Controls panel ───────────────────────────────────────────────────
        ctrl = tk.Frame(self.root, bg=_BG, pady=6)
        ctrl.grid(row=2, column=0, sticky='ew', padx=6)
        for col in range(10):
            ctrl.columnconfigure(col, weight=1)

        def _btn(parent, text, cmd, fg=_FG, bg=_BTN, bold=False):
            font = ('Arial', 10, 'bold') if bold else ('Arial', 10)
            return tk.Button(
                parent, text=text, command=cmd,
                bg=bg, fg=fg, relief='flat',
                activebackground='#666', activeforeground='white',
                font=font, padx=6, pady=5,
            )

        # ── Row 0 : trial navigation + annotation ────────────────────────────
        _btn(ctrl, '◀  Prev  [P]', self._prev_trial).grid(
            row=0, column=0, columnspan=2, sticky='ew', padx=2, pady=2)

        self.start_btn = _btn(ctrl, '● Mark Start  [S]', self._mark_start,
                              fg='white', bg='#2d6a4f', bold=True)
        self.start_btn.grid(row=0, column=2, columnspan=2, sticky='ew', padx=2, pady=2)

        self.end_btn = _btn(ctrl, '● Mark End  [E]', self._mark_end,
                            fg='white', bg='#7b2d2d', bold=True)
        self.end_btn.grid(row=0, column=4, columnspan=2, sticky='ew', padx=2, pady=2)

        _btn(ctrl, 'Save + Next  [N]  ▶', self._next_trial,
             fg='white', bg='#2d4a7b', bold=True).grid(
            row=0, column=6, columnspan=2, sticky='ew', padx=2, pady=2)

        _btn(ctrl, 'Skip  [X]', self._skip_trial,
             fg=_FG, bg='#5a4a1e').grid(
            row=0, column=8, columnspan=2, sticky='ew', padx=2, pady=2)

        # ── Row 1 : play / speed ─────────────────────────────────────────────
        self.play_btn = _btn(ctrl, '▶   Play   [Space]', self._toggle_play,
                             fg='white', bg='#3a5a3a', bold=True)
        self.play_btn.grid(row=1, column=0, columnspan=3, sticky='ew', padx=2, pady=2)

        speed_frame = tk.Frame(ctrl, bg=_BG3, padx=6, pady=4)
        speed_frame.grid(row=1, column=3, columnspan=7, sticky='ew', padx=2, pady=2)
        tk.Label(speed_frame, text='Speed:', bg=_BG3, fg='#aaa',
                 font=('Arial', 9)).pack(side='left', padx=(4, 8))

        self._speed_btns: dict = {}
        for sp in SPEED_OPTIONS:
            b = tk.Button(
                speed_frame, text=f'{sp}x',
                command=lambda s=sp: self._set_speed(s),
                bg=_BTN, fg=_FG, relief='flat',
                activebackground='#666', font=('Arial', 9), padx=8, pady=3,
            )
            b.pack(side='left', padx=2)
            self._speed_btns[sp] = b
        self._highlight_speed_btn(1.0)

        # ── Row 2 : seek slider ───────────────────────────────────────────────
        self.slider_var = tk.DoubleVar(value=0)
        self.slider = ttk.Scale(
            ctrl, from_=0, to=max(self.total_frames - 1, 1),
            orient='horizontal', variable=self.slider_var,
            command=self._on_slider_move,
        )
        self.slider.grid(row=2, column=0, columnspan=10, sticky='ew', padx=6, pady=4)
        self.slider.bind('<ButtonPress-1>',
                         lambda e: setattr(self, '_slider_dragging', True))
        self.slider.bind('<ButtonRelease-1>', self._on_slider_release)

        # ── Row 3 : annotation info bar ──────────────────────────────────────
        ann_bar = tk.Frame(ctrl, bg=_BG2, pady=6)
        ann_bar.grid(row=3, column=0, columnspan=10, sticky='ew', padx=2, pady=2)

        self.start_label = tk.Label(
            ann_bar, text='Start: —',
            bg=_BG2, fg=_GREEN, font=('Courier', 11, 'bold'))
        self.start_label.pack(side='left', padx=24)

        self.end_label = tk.Label(
            ann_bar, text='End: —',
            bg=_BG2, fg=_RED, font=('Courier', 11, 'bold'))
        self.end_label.pack(side='left', padx=24)

        self.trial_label = tk.Label(
            ann_bar, text='',
            bg=_BG2, fg=_YELLOW, font=('Courier', 11, 'bold'))
        self.trial_label.pack(side='right', padx=24)

    # =========================================================================
    # Key bindings
    # =========================================================================

    def _bind_keys(self):
        b = self.root.bind
        b('<space>',       lambda e: self._toggle_play())
        b('s',             lambda e: self._mark_start())
        b('S',             lambda e: self._mark_start())
        b('e',             lambda e: self._mark_end())
        b('E',             lambda e: self._mark_end())
        b('n',             lambda e: self._next_trial())
        b('N',             lambda e: self._next_trial())
        b('p',             lambda e: self._prev_trial())
        b('P',             lambda e: self._prev_trial())
        b('x',             lambda e: self._skip_trial())
        b('X',             lambda e: self._skip_trial())
        b('<Left>',        lambda e: self._step_frame(-1))
        b('<Right>',       lambda e: self._step_frame(1))
        b('<Shift-Left>',  lambda e: self._step_frame(-10))
        b('<Shift-Right>',  lambda e: self._step_frame(10))
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

    # =========================================================================
    # Playback loop
    # =========================================================================

    def _start_loop(self):
        self._tick()

    def _tick(self):
        if self.is_playing:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self._display_frame(frame)
            else:
                self.is_playing = False
                self.play_btn.config(text='▶   Play   [Space]', bg='#3a5a3a')

        if not self._slider_dragging:
            self.slider_var.set(self.current_frame)
        self._update_status()

        delay = max(1, int(1000 / (self.fps * self.speed)))
        self._after_id = self.root.after(delay, self._tick)

    def _display_frame(self, frame: np.ndarray):
        frame = self._draw_overlays(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((self.disp_w, self.disp_h), Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor='nw', image=self._photo)

    def _draw_overlays(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        font  = cv2.FONT_HERSHEY_SIMPLEX
        aa    = cv2.LINE_AA

        # Frame counter (top-left)
        cv2.putText(
            frame,
            f"Frame {self.current_frame}  |  {self.current_frame / self.fps:.2f}s",
            (12, 30), font, 0.65, (220, 220, 220), 1, aa,
        )

        # Speed indicator (top-right)
        cv2.putText(
            frame, f"{self.speed}x",
            (w - 68, 30), font, 0.65, (244, 162, 97), 2, aa,
        )

        # Start marker (bottom-left)
        if self.pending_start is not None:
            cv2.putText(
                frame,
                f"START: {self.pending_start}  ({self.pending_start / self.fps:.2f}s)",
                (12, h - 42), font, 0.60, (82, 183, 136), 2, aa,
            )
            if self.current_frame == self.pending_start:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (82, 183, 136), 5)

        # End marker
        if self.pending_end is not None:
            cv2.putText(
                frame,
                f"END:   {self.pending_end}  ({self.pending_end / self.fps:.2f}s)",
                (12, h - 16), font, 0.60, (224, 112, 112), 2, aa,
            )
            if self.current_frame == self.pending_end:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (224, 112, 112), 5)

        return frame

    # ── Seeking ──────────────────────────────────────────────────────────────

    def _seek(self, frame_num: int):
        frame_num = max(0, min(frame_num, self.total_frames - 1))
        self.current_frame = frame_num
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if ret:
            self._display_frame(frame)
        # Keep internal cap position at current_frame so next read() starts there
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        if not self._slider_dragging:
            self.slider_var.set(frame_num)
        self._update_status()

    def _step_frame(self, delta: int):
        self.is_playing = False
        self.play_btn.config(text='▶   Play   [Space]', bg='#3a5a3a')
        self._seek(self.current_frame + delta)

    def _toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.config(text='⏸   Pause   [Space]', bg='#5a3a1e')
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        else:
            self.play_btn.config(text='▶   Play   [Space]', bg='#3a5a3a')

    def _set_speed(self, speed: float):
        self.speed = speed
        self._highlight_speed_btn(speed)

    def _highlight_speed_btn(self, speed: float):
        for sp, btn in self._speed_btns.items():
            if sp == speed:
                btn.config(bg='#5b6da8', fg='white')
            else:
                btn.config(bg=_BTN, fg=_FG)

    # ── Slider ───────────────────────────────────────────────────────────────

    def _on_slider_move(self, val):
        pass   # actual seek handled on button release to avoid seek-storm

    def _on_slider_release(self, event):
        self._slider_dragging = False
        frame = int(self.slider_var.get())
        was_playing = self.is_playing
        self.is_playing = False
        self._seek(frame)
        if was_playing:
            self.is_playing = True
            self.play_btn.config(text='⏸   Pause   [Space]', bg='#5a3a1e')
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

    # =========================================================================
    # Trial navigation
    # =========================================================================

    def _load_current_trial(self):
        """Reset annotation state and seek to the current missed trial."""
        self.pending_start = None
        self.pending_end   = None
        self.start_btn.config(bg='#2d6a4f')
        self.end_btn.config(bg='#7b2d2d')
        self._update_annotation_labels()

        if self.missed_pos >= len(self.missed_indices):
            self.is_playing = False
            self.play_btn.config(text='▶   Play   [Space]', bg='#3a5a3a')
            messagebox.showinfo(
                "All Done",
                f"All {len(self.missed_indices)} missed trial(s) have been reviewed.\n"
                "Close the window to return the updated DataFrame.",
            )
            return

        row_idx = self.missed_indices[self.missed_pos]
        hint    = self._get_hint_frame(row_idx)
        if hint is not None:
            seek_to = max(0, hint - int(self.fps * 2))   # start 2 s before hint
            self._seek(seek_to)

        self._update_status()

    def _get_hint_frame(self, row_idx: int):
        """Return a frame number to seek to as a starting hint, or None."""
        row = self.trials_df.loc[row_idx]

        # Use trial_time — the window start where the actual start/stop must lie
        val = row.get('trial_time')
        if pd.notna(val):
            return int(val)

        return None

    # =========================================================================
    # Annotation actions
    # =========================================================================

    def _mark_start(self):
        self.pending_start = self.current_frame
        self.start_btn.config(bg='#1a8c5a')
        self._update_annotation_labels()

    def _mark_end(self):
        if self.pending_start is None:
            messagebox.showwarning("No Start Set",
                                   "Mark the start frame first (press S).")
            return
        if self.current_frame <= self.pending_start:
            messagebox.showwarning(
                "Invalid End Frame",
                f"End frame ({self.current_frame}) must come after "
                f"start frame ({self.pending_start}).",
            )
            return
        self.pending_end = self.current_frame
        self.end_btn.config(bg='#8c1a1a')
        self._update_annotation_labels()

    def _next_trial(self):
        if self.missed_pos >= len(self.missed_indices):
            return

        row_idx = self.missed_indices[self.missed_pos]

        if self.pending_start is not None and self.pending_end is not None:
            self._save_annotation(row_idx, self.pending_start, self.pending_end)
        else:
            missing = []
            if self.pending_start is None:
                missing.append('start')
            if self.pending_end is None:
                missing.append('end')
            if not messagebox.askyesno(
                "Incomplete Annotation",
                f"Missing: {' and '.join(missing)}.\n"
                "Skip this trial and leave it as NaN?",
            ):
                return

        self.missed_pos += 1
        self._load_current_trial()

    def _prev_trial(self):
        if self.missed_pos == 0:
            messagebox.showinfo("First Trial",
                                "Already at the first missed trial.")
            return

        self.missed_pos -= 1
        row_idx = self.missed_indices[self.missed_pos]

        # Revert any annotation we may have saved for this row
        self.trials_df.at[row_idx, 'start'] = np.nan
        self.trials_df.at[row_idx, 'stop']  = np.nan
        self.trials_df.at[row_idx, 'valid'] = 'no'

        self._load_current_trial()

    def _skip_trial(self):
        self.missed_pos += 1
        self._load_current_trial()

    def _save_annotation(self, row_idx: int, start: int, end: int):
        self.trials_df.at[row_idx, 'start']    = int(start)
        self.trials_df.at[row_idx, 'stop']     = int(end)
        self.trials_df.at[row_idx, 'duration'] = end - start
        self.trials_df.at[row_idx, 'valid']    = 'yes'

    # =========================================================================
    # UI label helpers
    # =========================================================================

    def _update_annotation_labels(self):
        if self.pending_start is not None:
            self.start_label.config(
                text=f"Start: {self.pending_start}  "
                     f"({self.pending_start / self.fps:.2f}s)")
        else:
            self.start_label.config(text='Start: —')

        if self.pending_end is not None:
            self.end_label.config(
                text=f"End: {self.pending_end}  "
                     f"({self.pending_end / self.fps:.2f}s)")
        else:
            self.end_label.config(text='End: —')

    def _update_status(self):
        n = len(self.missed_indices)
        pos = self.missed_pos + 1

        if self.missed_pos < n:
            row_idx = self.missed_indices[self.missed_pos]
            self.trial_label.config(
                text=f"Missed {pos} / {n}  (df row {row_idx})")
        else:
            self.trial_label.config(text="Complete ✓")

        self.status_var.set(
            f"  Frame {self.current_frame} / {self.total_frames - 1}"
            f"  |  {self.current_frame / self.fps:.2f}s"
            f"  |  {self.speed}x"
            f"  |  {'▶ PLAYING' if self.is_playing else '⏸ paused'}"
        )

    # =========================================================================
    # Cleanup
    # =========================================================================

    def _on_close(self):
        if self._after_id:
            self.root.after_cancel(self._after_id)
        self.is_playing = False
        self.cap.release()
        self.root.destroy()

    def get_result(self) -> pd.DataFrame:
        return self.trials_df


# =============================================================================
# Public API
# =============================================================================

def annotate_trials(
    video_path: str,
    trials_df: pd.DataFrame,
    hint_frame_col: str = None,
) -> pd.DataFrame:
    """
    Launch an interactive GUI for manually annotating missed trials.

    Opens a video player that iterates through every row in ``trials_df``
    where ``start_frame`` is NaN.  For each such trial, the video seeks to
    the approximate expected location (via ``window_start``) so the user
    can watch the footage and mark the exact start and end frame.

    Parameters
    ----------
    video_path : str
        Path to the experiment video file (.mp4, .avi, etc.).
    trials_df : pd.DataFrame
        DataFrame returned by ``validate_door_movements_in_windows``.  Must
        contain: ``trial_time``, ``end_time``, ``start``, ``stop``,
        ``duration``, ``valid`` ('yes'/'no').
    hint_frame_col : str, optional
        Override column for the seek hint frame.  If omitted, ``window_start``
        is used automatically.

    Returns
    -------
    pd.DataFrame
        A copy of ``trials_df`` with NaN values filled in for every trial
        the user annotated.  Skipped trials remain as NaN.

    Examples
    --------
    Basic usage after running extract_trial_times::

        from tmaze_toolkit.processing.extractTrialTimes import extract_trial_times
        from tmaze_toolkit.visualization.trial_annotator import annotate_trials

        # trials_df already has NaN rows for missed detections
        updated_df = annotate_trials(
            video_path=r'N:\\...\\session.mp4',
            trials_df=trials_df,
            hint_frame_col='floor_start_frame',
        )

    Keyboard shortcuts displayed in the GUI window title bar are also
    documented at the top of this module.
    """
    root = tk.Tk()
    root.resizable(False, False)

    app = TrialAnnotatorApp(root, video_path, trials_df, hint_frame_col)

    # Centre window on screen
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    ww, wh = root.winfo_width(),       root.winfo_height()
    root.geometry(f'+{(sw - ww) // 2}+{(sh - wh) // 2}')

    root.mainloop()
    return app.get_result()
