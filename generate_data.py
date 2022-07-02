from re import S
import pandas as pd
import numpy as np
from datetime import timedelta

def generate_synthetic_series() -> pd.DataFrame:
    seed = [
        {'date': '18/01/2021', 'sales': 4000, 'HoursWorked': 32},
        {'date': '19/01/2021', 'sales': 5000, 'HoursWorked': 32},
        {'date': '20/01/2021', 'sales': 3000, 'HoursWorked': 24},
        {'date': '21/01/2021', 'sales': 6000, 'HoursWorked': 40},
        {'date': '22/01/2021', 'sales': 10000, 'HoursWorked': 80},
        {'date': '23/01/2021', 'sales': 9000, 'HoursWorked': 80},
        {'date': '24/01/2021', 'sales': 6000, 'HoursWorked': 64}
    ]

    # add noise to the seed and create DataFrame
    for day in seed:
        day['sales'] += np.round(np.random.normal(scale=250),2)
        day['date'] = pd.to_datetime(day['date'])
        day['month'] = day['date'].month
        day['year'] = day['date'].year
        day['day_of_week'] = day['date'].dayofweek
        day['day_of_month'] = day['date'].day

    series = pd.DataFrame(seed)
    overshoot = False

    # add more data
    last_seed_date = series.iloc[-1]['date']
    for i in range(1, 1300):
        new_date = last_seed_date + timedelta(days=i)
        change_labour_index = np.random.choice(9, 1, replace=False)
        change_labour = np.array(np.linspace(-4, 4, 9))[change_labour_index]
        week_ago = new_date + timedelta(days=-7)
        row = series[series['date'] == week_ago]
        labor = row['HoursWorked'].values[0]
        sales = row['sales'].values[0]
        percent_change_labor = abs(change_labour)/labor
        labor += change_labour
        labor = max(8, labor)
        sales += percent_change_labor/2*sales*np.sign(change_labour)
        sales = max(0, sales + np.random.normal(scale=250))

        # every 10 days overshoot the labour
        if i % 4 == 0 and sales != 0:
            if not overshoot: 
                sales = 0.75*sales
            #else:
                #sales = 1.2*sales
                
            overshoot = not overshoot

        if sales == 0:
            sales = 500

        series = series.append({
            'date': new_date, 
            'year': new_date.year,
            'month': new_date.month,
            'day_of_week': new_date.dayofweek,
            'day_of_month': new_date.day,
            'HoursWorked': labor if isinstance(labor, int) else labor[0],
            'sales': np.round(sales if isinstance(sales, int) else sales[0], 2)
        }, ignore_index=True)

    series.index.name = 'index'

    for index, value in series.iterrows():
        if index < len(series)-1:
            series.loc[index, 'dependency_next'] = series.iloc[index+1]['HoursWorked']

    return series

if __name__ == '__main__':
    data = generate_synthetic_series()
    data.to_csv('generated_data_june.csv')