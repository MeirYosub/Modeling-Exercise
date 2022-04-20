import plotly.graph_objects as go
import numpy as np

def make_meshgrid(x, y, h=.01):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def make_decision_boundary_plot(clf_fit,X0,X1,x_grid,y_grid,title):
    Z = clf_fit.predict(np.c_[x_grid, y_grid])
    fig = go.Figure(data = go.Contour(x = x_grid, y = y_grid, 
                    z = Z,colorscale=[[0, 'rgb(204,229,255)'],[1,'rgb(255,153,150)']],showscale=False))
    fig=fig.add_trace(go.Scatter(x=X0,y=X1,
                    mode = 'markers',marker=dict(color=y,size=8,line=dict(color='black',width=1))))
    fig.update_layout(title=title,height=800)
    return fig