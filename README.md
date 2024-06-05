# 10day_buffer_scenario_generation
This repository contains the code used to implemented the proposed PCT method using 10-day buffer data in the thesis: Using 10-day buffer readouts for generating electricity consumption scenarios.
The code shows examples on how to use the proposed method aswell on how to train it, all images used in the thesis can also be generated using this repository.

It is important to note that the following repository by Jonas Soenen for the original preprocessing https://github.com/jonassoenen/predclus_scengen

# The datasets
In total 4 datasets are being used
London: https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=7857
Ireland: https://www.ucd.ie/issda/data/commissionforenergyregulationcer/
Flanders: In the thesis referred to as Flanders2021, download no longer available
FlandersBig: In the thesis referred to as Flanders2022 https://opendata.fluvius.be/explore/dataset/1_50-verbruiksprofielen-dm-elek-kwartierwaarden-voor-een-volledig-jaar/information/

# How to get started
- Retrieve all the required packages by creating a new conda environment or similar using the environment.yml file.
- Run the preprocessing files (notebooks in the data_preprocessing folder) or by using the preprocessing done by J. Soenen.
- Retrieve the buffer attributes (get_buffer_attributes... in the data_preprocessing folder).
- Train the models using the ModelTrainer... notebooks in the notebooks folder.
- Extra visualisations can be seen in the Visualise... notebooks in the notebooks folder but are not required, they can visualize the PCTs as well as the feature importances.

# Additional
- The files follow a consistent naming pattern, of no indication of a dataset is given, it is using the London dataset, anything including something related to Ireland is using the Ireland dataset, for the Flanders2022 dataset FlandersBig is used, the other files just mentioning Flanders refer to the Flanders2021 dataset that is no longer publicly available.
- Whenever a path is given to store or retrieve something, you should dubble check that you use the same naming structure and might have to add the root path.


Furthermore, I wish those brave enough to embark on this journey the best of luck and hope they will share in my enthusiasm when seeing the results!
