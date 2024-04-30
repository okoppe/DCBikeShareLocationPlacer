# DC Bike Share Location Placer
To run this code, first go to chpc and start a new Spark Jupyter Session. Then transfer all the files from this repository to chpc and open the repository.

To begin, open the bike_share_data folder. In this folder open the mergeBikeShareData.ipynb file. Notice at the top of the notebook there are pip install statements. Run the first cell to install the packages. Then restart the kernel, if you do not restart the kernel the code will not run as the packages are not available until after restarting the kernel.

Run all cells in the mergeBikeShareData.ipynb. This will create one large merged data set for all the rideshare rides.

Next open the featureEngineeringDataCleaning.ipynb file in the base folder. Again, run the first cell to install the packages. Then restart the kernel, if you do not restart the kernel the code will not run as the packages are not available until after restarting the kernel.

After restarting the kernel, run all code cells in the notebook. This notebook uses spark to clean and aggregate the data into a small data set with features for each ride share location.

Finally, open the map.ipynb file and run the first cell to install the packages. Then restart the kernel, if you do not restart the kernel the code will not run as the packages are not available until after restarting the kernel.

Then run all cells in this kernel. In the output of the last cell you will see two widgets and a submit button. Adjust the values in the widget boxes as you see fit and then hit submit. After hitting submit you will see which stations were added and removed, some evaluation metrics, and an interactive plot of the regions each bike share location serves.

Note, running the same values for Add stations and Remove may yield different results as there is k means clustering going on in the background on randomly generated points. To keep run times somewhat fast we choose a medium number of random points. The more random points generated the more stable results are. See our write up for more info.
