import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

data = pd.read_csv("trainingset.csv")

fig1 = plt.figure('feature1')
plt.suptitle("# of cars from each manufacturer")
plt.xlabel("Make model")
make, counts = np.unique(data.loc[:, 'feature1'], return_counts=True)
plt.bar(make, counts)

fig2 = plt.figure('feature2')
plt.suptitle("# of cars with fuel type")
plt.xlabel("Fuel type")
fuel_types, counts = np.unique(data.loc[:, 'feature2'], return_counts=True)
plt.bar(fuel_types, counts)

fig3 = plt.figure('feature3')
plt.suptitle("# of cars with type of aspiration")
plt.xlabel("Aspiration")
aspiration, counts = np.unique(data.loc[:, 'feature3'], return_counts=True)
plt.bar(aspiration, counts)

fig4 = plt.figure('feature4')
plt.suptitle("# of cars with body type")
plt.xlabel("Body style")
body_style, counts = np.unique(data.loc[:, 'feature4'], return_counts=True)
plt.bar(body_style, counts)

fig5 = plt.figure('feature5')
plt.suptitle("# of doors on cars")
plt.xlabel("# of doors")
door_num, counts = np.unique(data.loc[:, 'feature5'], return_counts=True)
plt.bar(door_num, counts)

fig6 = plt.figure('feature6')
plt.suptitle("Side of steering wheel")
plt.xlabel("drive wheel side")
drive_wheel, counts = np.unique(data.loc[:, 'feature6'], return_counts=True)
plt.bar(drive_wheel, counts)

fig7 = plt.figure('feature7')
plt.suptitle("Engine location")
plt.xlabel("Location")
engine_location, counts = np.unique(data.loc[:, 'feature7'], return_counts=True)
plt.bar(engine_location, counts)

fig8 = plt.figure('feature8')
plt.suptitle("Engine types")
plt.xlabel("Engine type")
engine_type, counts = np.unique(data.loc[:, 'feature8'], return_counts=True)
plt.bar(engine_type, counts)

fig9 = plt.figure('feature9')
plt.suptitle("# of cylinders on cars")
plt.xlabel("# of cylinders")
num_of_cylinders, counts = np.unique(data.loc[:, 'feature9'], return_counts=True)
plt.bar(num_of_cylinders, counts)

fig10 = plt.figure('feature10')
plt.suptitle("Fuel systems on cars")
plt.xlabel("Fuel systems")
fuel_system, counts = np.unique(data.loc[:, 'feature10'], return_counts=True)
plt.bar(fuel_system, counts)

fig11 = plt.figure('feature11')
plt.suptitle("Fuel systems on cars")
plt.xlabel("Fuel systems")
fuel_system, counts = np.unique(data.loc[:, 'feature11'], return_counts=True)
plt.bar(fuel_system, counts)

fig12 = plt.figure('feature11')
plt.suptitle("Fuel systems on cars")
plt.xlabel("Fuel systems")
fuel_system, counts = np.unique(data.loc[:, 'feature11'], return_counts=True)
plt.bar(fuel_system, counts)

fig13 = plt.figure('feature12')
plt.suptitle("Fuel systems on cars")
plt.xlabel("Fuel systems")
fuel_system, counts = np.unique(data.loc[:, 'feature12'], return_counts=True)
plt.bar(fuel_system, counts)

fig14 = plt.figure('feature13')
plt.suptitle("Fuel systems on cars")
plt.xlabel("Fuel systems")
fuel_system, counts = np.unique(data.loc[:, 'feature13'], return_counts=True)
plt.bar(fuel_system, counts)

fig15 = plt.figure('feature14')
plt.suptitle("Fuel systems on cars")
plt.xlabel("Fuel systems")
fuel_system, counts = np.unique(data.loc[:, 'feature14'], return_counts=True)
plt.bar(fuel_system, counts)

fig16 = plt.figure('feature15')
plt.suptitle("Fuel systems on cars")
plt.xlabel("Fuel systems")
fuel_system, counts = np.unique(data.loc[:, 'feature15'], return_counts=True)
plt.bar(fuel_system, counts)

fig17 = plt.figure('feature16')
plt.suptitle("Fuel systems on cars")
plt.xlabel("Fuel systems")
fuel_system, counts = np.unique(data.loc[:, 'feature16'], return_counts=True)
plt.bar(fuel_system, counts)

fig18 = plt.figure('feature17')
plt.suptitle("Fuel systems on cars")
plt.xlabel("Fuel systems")
fuel_system, counts = np.unique(data.loc[:, 'feature17'], return_counts=True)
plt.bar(fuel_system, counts)

fig19 = plt.figure('feature18')
plt.suptitle("Fuel systems on cars")
plt.xlabel("Fuel systems")
fuel_system, counts = np.unique(data.loc[:, 'feature18'], return_counts=True)
plt.bar(fuel_system, counts)

plt.show()
