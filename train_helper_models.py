from argparse import ArgumentParser
from forecasting_models.prophet_model import ProphetModel

def parse_command_args():
    parser = ArgumentParser()

    parser.add_argument("-dp", "--data_path", dest="data_path",required=True,
                        help="specify the path to the data file stored in CSV format")

    args = parser.parse_args()
    return args

def train_forecast_sales(args):
    location_id = 14922

    forecast_sales_model = ProphetModel(args.data_path, 
                                        location_id=location_id,
                                        regressor="HoursWorked")
    forecast_sales_model.fit()
    forecast_sales_model.test_predict(title_file="metrics_sales_with_regressor.csv")
    forecast_sales_model.save_model("sales_model")

def train_forecast_hours(args):
    location_id = 14922

    forecast_hours_model = ProphetModel(args.data_path,
                                        output="HoursWorked",
                                        regressor="sales",
                                        location_id=location_id)

    forecast_hours_model.fit()
    forecast_hours_model.test_predict(title_file="metrics_hours_with_regressor.csv")
    forecast_hours_model.save_model("hours_model")

if __name__ == '__main__':
    args = parse_command_args()
    train_forecast_sales(args)
    train_forecast_hours(args)