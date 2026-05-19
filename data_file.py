import pandas as pd

data = pd.read_csv('FMF_data2.csv')

data['prevalence_0_49'] = (data['Registered_0-49'] / data['0-49 population_Total']).round(6)

data.to_csv('FMF_data2.csv', index=False)

print(data[['years', 'Registered_0-49', 'prevalence_0_49']])