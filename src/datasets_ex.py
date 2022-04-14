from sklearn.datasets import make_circles, make_moons
import pandas as pd

def make_circles_dataframe(n_samples, noise_level):
    points, label = make_circles(n_samples=n_samples, noise=noise_level)
    circles_df = pd.DataFrame(points, columns=['x','y'])
    circles_df['label'] = label
    circles_df.label = circles_df.label.map({0:'A', 1:'B'})
    return circles_df


def make_moons_dataframe(n_samples, noise_level):
    points, label = make_moons(n_samples=n_samples, noise=noise_level)
    moons_df = pd.DataFrame(points, columns=['x','y'])
    moons_df['label'] = label
    moons_df.label = moons_df.label.map({0:'A', 1:'B'})
    return moons_df

def make_exercise_dataframe():
    df=pd.DataFrame()
    nl_range=[nl/10 for nl in range(6)] 

    for nl in nl_range:
        df1=make_circles_dataframe(n_samples=10000, noise_level=nl)
        df1['dataset_name']='circles'
        df1['noise_level']=nl
        cols = df1.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        df1=df1[cols]
        df=pd.concat([df,df1],ignore_index=True)
    
    for nl in nl_range:
        df1=make_moons_dataframe(n_samples=10000, noise_level=nl)
        df1['dataset_name']='moons'
        df1['noise_level']=nl
        cols = df1.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        df1=df1[cols]
        df=pd.concat([df,df1],ignore_index=True)
        
    return df
