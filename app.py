import streamlit as st
import numpy as np
import pandas as pd

st.title("Variogram")
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

# from unicodedata import east_asian_width
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.metrics import pairwise_distances

# def uTransform(u,alpha, beta):
#     """
#     Returns the tranformed u
#     params:
#         u: input u is a numpy array of size n x 2
#         alpha: the rotation parameter
#         beta: the stretch parameter
#     returns:
#         u_t: the transformed u
#     """
#     assert u.shape[1] == 2, "u should be a numpy array of size n x 2"
#     assert isinstance(alpha, (int, float)), "alpha should be a float or integer"
#     assert isinstance(beta, (int, float)), "beta should be a float or integer"
#     R = np.array([
#         [np.cos(alpha), -np.sin(alpha)],
#         [np.sin(alpha), np.cos(alpha)]
#     ])
#     T = np.array([
#         [beta, 0],
#         [0, 1]
#     ])

#     u_t = T @ R @ u.T
#     return u_t.T

# def semivariogram(u,y):
#     """
#     Calculates the variogram of the data.
#     params:
#         u: numpy array of shape (n,2)
#         y: numpy array of shape (n,)
#     returns:
#         results: numpy array of shape (n,2)
#     """
#     assert u.shape[1] == 2, "u should be a numpy array of size n x 2"
#     assert y.shape[0] == u.shape[0], "u and y should have the same number of rows"
#     D = pairwise_distances(u, u)
#     lags = np.linspace(0, D.max()*0.6, 60)
#     vario = []
#     d = lags[1] - lags[0]
#     for l in lags:
#         ind = np.where((D>l-0.5*d) & (D<l+0.5*d))
#         s = np.sqrt(np.square(y[ind[0]] - y[ind[1]]).mean())
#         vario.append([l,s])
#     return np.array(vario)

# df = pd.read_csv('./data.csv')
# df = df[0:200]
# u = df[['x', 'y']].to_numpy()
# y = df['z'].to_numpy()

# v = semivariogram(u,y)
# alpha = 1
# beta = 1
# uT = uTransform(u,alpha, beta)

# dfV = pd.DataFrame(v, columns=['lag', 'semivariance'])
# # make lag be the index
# dfV.set_index('lag', inplace=True)
# st.line_chart(dfV)