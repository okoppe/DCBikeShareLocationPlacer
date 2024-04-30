import pandas as pd
import pointpats
import numpy as np
from shapely import intersection
from shapely.geometry import MultiPoint, Polygon, Point
from shapely import voronoi_polygons

class weightedVoroniDigram:
    """
    Class that computes weighted voronoi digrams for our use case
    
    """
    def find_optimal_voronoi_digram(num_to_add, num_to_remove, df, boundary):
        """
        Args:
            num_to_add (int): the number of points to add to data
            num_to_remove (int): number of points in dataset to remove
            df (dataframe): dataframe with a column for "weights", "station_name", and the "point" (x,y).
            boundary (shapley polygon): Bound the Vorinoi digram to this region.
        """
        df = df[['station_name', 'LATITUDE', 'LONGITUDE', 'point','weight']]

        removed_df = weightedVoroniDigram.__find_optimal_point_to_remove(num_to_remove, df, boundary)
        added_removed_df = weightedVoroniDigram.__find_optimal_points_to_add(num_to_add, removed_df, boundary)
            
        return added_removed_df

    def evaluate(original_data, new_data, boundary):
        original_vorregions = weightedVoroniDigram.voronoi_polygons_and_weights(original_data, boundary)
        new_vorregions = weightedVoroniDigram.voronoi_polygons_and_weights(new_data, boundary)

        # average vor region size
        old_areas = []
        for index, row in original_vorregions.iterrows():
            old_areas.append(Polygon(row['poly']).area)
        original_vorregions['area'] = old_areas
        
        new_areas = []
        for index, row in new_vorregions.iterrows():
            new_areas.append(Polygon(row['poly']).area)
        new_vorregions['area'] = new_areas
        print("Evaluation metrics")
        print(f"Original station average voronoi region size: {original_vorregions['area'].mean()}")
        print(f"New station average voronoi region size: {new_vorregions['area'].mean()}")

        print(f"Original station median voronoi region size: {original_vorregions['area'].median()}")
        print(f"New station median voronoi region size: {new_vorregions['area'].median()}")    

    
    def __find_optimal_point_to_remove(num_to_remove, df, boundary):
        """
        Args:
            num_to_remove (int): number of points in dataset to remove
            df (dataframe): dataframe with a column for "weights", "station_name", and the "point" (x,y).
            boundary (shapley polygon): Bound the Vorinoi digram to this region.

        return, dataframe of with the num_to_remove points removed that were the worst locations
        """
        i = 0
        while i<num_to_remove:
            df = weightedVoroniDigram.voronoi_polygons_and_weights(df, boundary)
            print(f"Removing Station {df.iloc[-1]['station_name']} at {df.iloc[-1]['LATITUDE']}, {df.iloc[-1]['LONGITUDE']}")
            # remove the smallest weighted region
            df.drop(df.index[-1], axis=0, inplace=True)
            i=i+1
        
        return df
        
    def __find_optimal_points_to_add(num_to_add, df, boundary):
        """
        Probably use regression to compute the anticipated weights for the new points to be added.

        
        Args:
            num_to_add (int): the number of points to add to data
            df (dataframe): dataframe with a column for "weights", "station_name", and the "point" (x,y).
            boundary (shapley polygon): Bound the Vorinoi digram to this region.

        return, dataframe of with the num_to_add points added that were the best locations
        """
        i = 0
        new_stations = []
        while i<num_to_add:
            df = weightedVoroniDigram.voronoi_polygons_and_weights(df, boundary)

            largest_weigthed_region = df.iloc[0]
            # generate 10,000 random points inside the largest poly
            pts = pointpats.random.poisson(Polygon(largest_weigthed_region['poly']), size=10000)

            # do kmeans
            kmeans = KMeansOneFixedCenter(np.array([largest_weigthed_region['LATITUDE'], largest_weigthed_region['LONGITUDE']]))
            
            # Fit the model to the data
            kmeans.fit(pts)
            kmeans.centroids
            
            # drop the original col for two new ones
            df = df[df['station_name']!=largest_weigthed_region['station_name']]
            
            # new center
            new_stations.append([kmeans.centroids[0][0], kmeans.centroids[0][1]])
            df2 = pd.DataFrame([{'station_name': f"new station {i}", 'LATITUDE': kmeans.centroids[0][0], 'LONGITUDE':kmeans.centroids[0][1], 
                                'point': Point(kmeans.centroids[0]),'poly': 0, 'weight': largest_weigthed_region['weight']/2, 'poly_weight': 0}])
            df3 = pd.DataFrame([{'station_name': largest_weigthed_region['station_name'], 'LATITUDE': largest_weigthed_region['LATITUDE'], 
                                 'LONGITUDE':largest_weigthed_region['LONGITUDE'],'point': largest_weigthed_region['point'],'poly': 0, 'weight': largest_weigthed_region['weight']/2, 'poly_weight': 0}])
            df = pd.concat([df3, df2, df])
            i = i+1
        print(f"New stations at {[x for x in new_stations]}") 
        df = weightedVoroniDigram.voronoi_polygons_and_weights(df, boundary)

        return df
        
    def voronoi_polygons_and_weights(df, boundary):
        """
        Given a dataframe with weights compute the shapley poloygons for the voronoi digram on the pts and compute the weight for each point.
        
        Args:
            df (dataframe): dataframe with a column for "weights", "station_name", and the "point" (x,y)
            boundary (shapley polygon): Bound the Vorinoi digram to this region.
        Returns:
            dataframe: with the columns "poly", "poly_weight", column of Shapley Poloygons that is the voronoi digram for the point, 
                                                    "weights", "station_name", and the "point" (x,y)
        """
        points2 =  MultiPoint(df['point'].to_list())

        voronoi_diagram = voronoi_polygons(points2)

        vor_polygons = []
        for geom in voronoi_diagram.geoms:
            xx, yy = geom.exterior.coords.xy
            coord = list(zip(xx, yy))
            inter = intersection(geom, boundary)
            xx, yy = inter.exterior.coords.xy
            coord_clipped = list(zip(xx, yy))
            
            vor_polygons.append(coord_clipped)

        df['poly'] = vor_polygons

        poly_w = []

        for index, row in df.iterrows():
            poly_w.append(Polygon(row['poly']).area*row['weight'])

        df['poly_weight'] = poly_w
        df.sort_values(by='poly_weight', ascending=False, inplace=True)
        
        return df


class KMeansOneFixedCenter:
    def __init__(self, fixedCenter, max_iters=100):
        n_clusters=2
        self.fixedCenter = fixedCenter
        self.n_clusters = n_clusters
        self.max_iters = max_iters

    def fit(self, X):
        # Initialize centroids randomly
        
        self.centroids = np.concatenate((X[np.random.choice(X.shape[0], 1, replace=False)], [self.fixedCenter]))
        
        for _ in range(self.max_iters):
            # Assign each data point to the nearest centroid
            labels = self._assign_labels(X)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check for convergence
            if np.all(self.centroids == new_centroids):
                break
                
            self.centroids = new_centroids

    def _assign_labels(self, X):
        # Compute distances from each data point to centroids
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        
        # Assign labels based on the nearest centroid
        v = np.argmin(distances, axis=1)
        return v
    
    def _update_centroids(self, X, labels):
        new_centroids = []
        for i in range(self.n_clusters):
            if i == 1:
                new_centroids.append(self.fixedCenter)
            else:
                new_centroids.append(X[labels == i].mean(axis=0))
                
        return np.array(new_centroids)