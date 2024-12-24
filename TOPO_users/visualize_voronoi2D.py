import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d


if __name__ == "__main__":
    # Generate random points
    points = np.random.rand(10, 2)

    # Compute Voronoi diagram
    vor = Voronoi(points)

    # Plot Voronoi diagram
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor,
                    ax=ax,
                    show_points=True,
                    show_vertices=True,
                    line_colors='black',
                    line_width=2,
                    line_alpha=1,
                    point_size=16)

    # Highlight a central point
    # central_point = points  # Select the 6th point as central
    # ax.plot(central_point[0], central_point[1], 'ro')  # Blue point

    # Plot the vertices
    # vertex_point = vor.vertices
    # ax.plot(vertex_point[0], vertex_point[1], 'go')  # Green points

    # Set plot limits
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    ax.axis('off')

    plt.show()
    print('haha')
