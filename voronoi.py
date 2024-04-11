import pandas as pd
from shapely import intersection
from shapely.geometry import MultiPoint, Polygon, Point
from shapely import voronoi_polygons

class weightedVoroniDigram:
    """
    Class that computes weighted voronoi digrams for our use case
    
    """
    def __init__(self, df):
        """
        """
        self.data = df

    def find_optimal_voronoi_digram(num_to_add, num_to_remove, df, boundary):
        """
        Args:
            num_to_add (int): the number of points to add to data
            num_to_remove (int): number of points in dataset to remove
            df (dataframe): dataframe with a column for "weights", "station_name", and the "point" (x,y).
            boundary (shapley polygon): Bound the Vorinoi digram to this region.
        """
        
        removed_df = weightedVoroniDigram.__find_optimal_point_to_remove(num_to_remove, df, boundary)
        added_removed_df = weightedVoroniDigram.__find_optimal_points_to_add(num_to_add, removed_df, boundary)

        
        
            
        return weightedVoroniDigram.__weighted_voronoi_polygons(added_removed_df, boundary)
        

    def __find_optimal_point_to_remove(num_to_remove, df, boundary):
        """
        Args:
            num_to_remove (int): number of points in dataset to remove
            df (dataframe): dataframe with a column for "weights", "station_name", and the "point" (x,y).
            boundary (shapley polygon): Bound the Vorinoi digram to this region.

        return, dataframe of with the num_to_remove points removed that were the worst locations
        """
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
        return df
        
    def __weighted_voronoi_polygons(df, boundary):
        """
        Given a dataframe with weights compute the shapley poloygons for the weighted voronoi digram on the pts.
        
        Args:
            df (dataframe): dataframe with a column for "weights", "station_name", and the "point" (x,y)
            boundary (shapley polygon): Bound the Vorinoi digram to this region.
        Returns:
            dataframe: with the columns "poly", column of Shapley Poloygons that is the voronoi digram for the point, 
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
        return df

