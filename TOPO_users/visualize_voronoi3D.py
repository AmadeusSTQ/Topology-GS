import numpy as np
from scipy.spatial import Delaunay
from mayavi import mlab
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# 生成球面上的点
def fibonacci_sphere(samples=100):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # 黄金角
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # 从1到-1
        radius = np.sqrt(1 - y * y)  # 半径在x-z平面上的投影
        theta = phi * i  # 黄金角度
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)


if __name__ == "__main__":
    # method = 'mayavi'
    # method = 'plotly'
    method = 'matplotlib'

    # 生成球面上的点
    points = fibonacci_sphere(samples=48)

    # 使用 Delaunay 进行三角剖分生成三角形
    tri = Delaunay(points[:, :2])  # 注意，这里我们只使用前两个维度进行三角剖分

    # 获取三角形的顶点索引
    triangles = points[tri.simplices]

    # 计算三角形中心
    centroids = np.mean(triangles, axis=1)

    if method == 'mayavi':

        # 创建3D绘图
        mlab.figure(bgcolor=(1, 1, 1))

        # 绘制三角形
        for triangle in triangles:
            tri = mlab.triangular_mesh(
                [triangle[0][0], triangle[1][0], triangle[2][0]],
                [triangle[0][1], triangle[1][1], triangle[2][1]],
                [triangle[0][2], triangle[1][2], triangle[2][2]],
                [[0, 1, 2]],
                color=(0.5, 0.5, 0.5),
                opacity=0.3
            )

        # 绘制三角形的中心
        mlab.points3d(
            centroids[:, 0], centroids[:, 1], centroids[:, 2],
            color=(1, 0, 0),
            scale_factor=0.05
        )

        # 绘制三角形的顶点
        mlab.points3d(
            points[:, 0], points[:, 1], points[:, 2],
            color=(0, 0, 1),
            scale_factor=0.05
        )

        # 设置视角
        mlab.view(azimuth=45, elevation=60, distance=10)

        # 保存图像
        # mlab.savefig('tetrahedrons_on_sphere.png')
        mlab.show()

    elif method == 'plotly':
        # 绘制三角形
        fig = go.Figure()

        for triangle in triangles:
            x, y, z = triangle[:, 0], triangle[:, 1], triangle[:, 2]
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=[0], j=[1], k=[2],
                color='gray',
                opacity=0.3,
                flatshading=True
            ))

        # 绘制三角形的中心
        fig.add_trace(go.Scatter3d(
            x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2],
            mode='markers',
            marker=dict(size=4, color='red'),
            name='Centers'
        ))

        # 绘制三角形的顶点
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=4, color='blue'),
            name='Vertices'
        ))

        # 更新布局
        fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False)
            ),
            title='Triangles on a Sphere with Centers and Vertices',
            showlegend=True
        )

        # 显示图形
        fig.show()

    else:
        # 创建绘图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        u = np.linspace(0, 2*np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='green', alpha=0.1)

        # 绘制三角形
        for triangle in triangles:
            verts = [triangle]
            poly = Poly3DCollection(verts, alpha=0.3, facecolor='gray', edgecolor='k')
            ax.add_collection3d(poly)

        # 绘制三角形的中心
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], color='b', label='Centers')

        # 绘制三角形的顶点
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='orange', label='Vertices')

        # 设置图形标签
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        ax.set_axis_off()
        # ax.set_title('Triangles on a Sphere with Centers and Vertices')

        # 设置显示范围
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        # 显示图例
        # ax.legend()

        # 显示图形
        plt.show()
