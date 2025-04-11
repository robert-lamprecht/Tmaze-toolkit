import matplotlib.pyplot as plt

def plotDoorTraces(data):
    # Create a figure with subplots for each trace
    numPlots = len(data.keys())
    fig, axs = plt.subplots(numPlots, 1, figsize=(15, 10), sharex=True)
    fig.suptitle('Door Traces')

    # Plot each trace
    for i, (door, trace) in enumerate(data.items()):
        axs[i].plot(trace)
        axs[i].set_title(f'{door} Trace')
        axs[i].set_ylabel('Motion')

    axs[-1].set_xlabel('Frame Number')
    plt.tight_layout()
    plt.show()
