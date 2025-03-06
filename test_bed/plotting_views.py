from graspnetAPI.utils import utils
import plotly.express as px
import pandas as pd

# Generate views
views = utils.generate_views(300)

# Assuming views is a list of tuples or a 2D array with (x, y, z) coordinates
# Convert views to a DataFrame for Plotly Express

df = pd.DataFrame(views, columns=['x', 'y', 'z'])

# Create a 3D scatter plot
fig = px.scatter_3d(df, x='x', y='y', z='z', title='3D Scatter Plot of Views')
fig.show()

