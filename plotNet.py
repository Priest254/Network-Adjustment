import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from itertools import combinations


# Results directory
result_dir = "C:\\Users\\Allan\\Documents\\Year 5 Project\\Results"
def plotNet(df_coord, s ):
    s = s.reshape(-1,2)  # reshape Standard deviations
    # Plot the points
    fig_size = (16, 12)
    fig, ax= plt.subplots(figsize=fig_size, dpi=700)
    plt.scatter(df_coord["Y"], df_coord["X"], color='red')
    
    # Annotate each point with its station name
    for i, point in df_coord.iterrows():
        plt.annotate(point['STATION'], (point['Y'], point['X']), textcoords="offset points", xytext=(0,10), ha='center')
    
    # Draw lines from each station to every other station
    for i, point1 in df_coord.iterrows():
        for j, point2 in df_coord.iterrows():
            if i != j:
                plt.plot([point1['Y'], point2['Y']], [point1['X'], point2['X']], 'b-', alpha=0.3)
    
    # Add error ellipses
    for i, row in df_coord.iterrows():
        ellipse = Ellipse((row['Y'], row['X']), width=s[i,0], height=s[i,1], edgecolor='g', fc='None', lw=2)
        ax.add_patch(ellipse)
        
    
    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('FREE NET 001')
    
    # Add grid
    plt.grid(True)
    
    # Save plot as JPEG
    #plt.savefig("FREE NET Final.jpg", format='jpeg',dpi = 700)
    # Show plot
    plt.show()
