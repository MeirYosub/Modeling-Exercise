import plotly.graph_objects as go
import numpy as np

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def make_decision_boundary_plot(clf_fit,X0,X1,y,x_grid,y_grid,title):
    Z = clf_fit.predict(np.c_[x_grid, y_grid])
    fig = go.Figure(data = go.Contour(x = x_grid, y = y_grid, 
                    z = Z,colorscale=[[0, 'rgb(204,229,255)'],[1,'rgb(255,153,150)']],showscale=False))
    
    
    fig=fig.add_trace(go.Scatter(name='Class 0',x=X0[y==0],y=X1[y==0],
                    mode = 'markers',marker=dict(color='Blue',size=8,line=dict(color='black',width=1))))
    fig=fig.add_trace(go.Scatter(name='Class 1',x=X0[y==1],y=X1[y==1],
                    mode = 'markers',marker=dict(color='Red',size=8,line=dict(color='black',width=1))))
    
    fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))
    fig.update_layout(title=title,width=1000,height=800)
    return fig