import numpy as np
import joblib as jb
import pandas as pd

data_plz = pd.read_parquet(
    "https://github.com/Immobilienrechner-Challenge/data/blob/main/plz_data.parquet?raw=true"
)
scaler = jb.load("model1/utils/scaler_histgradientboostingregressor.joblib")
model = jb.load("model1/utils/model_histgradientboostingregressor.joblib")
cols = [
    "floor_space",
    "last_refurbishment",
    "living_space",
    "plot_area",
    "rooms",
    "year_built",
    "zip_code",
    "type_attic-flat",
    "type_attic-room",
    "type_castle",
    "type_chalet",
    "type_detached-house",
    "type_detached-secondary-suite",
    "type_duplex-maisonette",
    "type_farmhouse",
    "type_flat",
    "type_furnished-residential-property",
    "type_loft",
    "type_penthouse",
    "type_rustico",
    "type_secondary-suite",
    "type_semi-detached-house",
    "type_single-room",
    "type_stepped-apartment",
    "type_stepped-house",
    "type_studio",
    "type_terrace-house",
    "type_villa",
    "gde_average_house_hold",
    "NoisePollutionRoadS",
    "PopulationDensityM",
    "NoisePollutionRoadL",
    "gde_workers_sector1",
    "ForestDensityL",
    "ForestDensityM",
    "gde_tax",
    "gde_politics_sp",
    "gde_social_help_quota",
    "RiversAndLakesS",
    "NoisePollutionRoadM",
    "gde_private_apartments",
    "ForestDensityS",
    "distanceToTrainStation",
    "NoisePollutionRailwayM",
    "gde_workers_total",
    "gde_workers_sector3",
    "PopulationDensityL",
    "gde_area_forest_percentage",
    "gde_politics_pda",
    "gde_politics_bdp",
    "RiversAndLakesM",
    "RiversAndLakesL",
    "NoisePollutionRailwayL",
    "gde_politics_evp",
    "WorkplaceDensityM",
    "gde_area_nonproductive_percentage",
    "gde_foreigners_percentage",
    "gde_politics_glp",
    "gde_pop_per_km2",
    "gde_politics_cvp",
    "gde_population",
    "WorkplaceDensityL",
    "Longitude",
    "gde_area_settlement_percentage",
    "NoisePollutionRailwayS",
    "gde_new_homes_per_1000",
    "gde_empty_apartments",
    "WorkplaceDensityS",
    "gde_area_agriculture_percentage",
    "gde_politics_rights",
    "gde_politics_fdp",
    "PopulationDensityS",
    "gde_politics_gps",
    "gde_workers_sector2",
    "Latitude",
    "gde_politics_svp",
]
types = [
    "attic-flat",
    "attic-room",
    "castle",
    "chalet",
    "detached-house",
    "detached-secondary-suite",
    "duplex-maisonette",
    "farmhouse",
    "flat",
    "furnished-residential-property",
    "loft",
    "penthouse",
    "rustico",
    "secondary-suite",
    "semi-detached-house",
    "single-room",
    "stepped-apartment",
    "stepped-house",
    "studio",
    "terrace-house",
    "villa",
]

# Create your models here.
class PredictionHistGradientBoostingRegression:
    def __init__(
        self,
        living_space=None,
        floor_space=None,
        type=None,
        plot_area=None,
        rooms=None,
        zip_code=None,
        last_refurbishment=None,
        year_built=None,
    ):
        # set attributes with correct type
        self.living_space = np.float(living_space)
        self.floor_space = np.float(floor_space)
        self.type = type
        self.plot_area = np.float(plot_area)
        self.rooms = np.float(rooms)
        self.zip_code = int(zip_code)
        self.last_refurbishment = np.float(last_refurbishment)
        self.year_built = np.float(year_built)

    def __generateDataFrame(self, cols, data_plz):
        # create empty dataframe with columns
        df = pd.DataFrame(columns=cols, index=[0])
        # fill dataframe with values
        if not np.isnan(self.living_space):
            df["living_space"] = self.living_space
        if not np.isnan(self.floor_space):
            df["floor_space"] = self.floor_space
        if not np.isnan(self.plot_area):
            df["plot_area"] = self.plot_area
        if not np.isnan(self.rooms):
            df["rooms"] = self.rooms
        if not np.isnan(self.last_refurbishment):
            df["last_refurbishment"] = self.last_refurbishment
        if not np.isnan(self.year_built):
            df["year_built"] = self.year_built
        # special case for type
        if self.type:
            for col in df.columns:
                if "type_" in col:
                    df[col] = 0
            df[f"type_{self.type}"] = 1
        # special case for zip_code
        if not np.isnan(self.zip_code):
            df["zip_code"] = self.zip_code
            df[data_plz.columns] = data_plz.loc[self.zip_code]
        # return dataframe
        return df

    def __scaleDataFrame(self, df, cols, scaler):
        # scale dataframe
        return pd.DataFrame(scaler.transform(df[cols]), columns=cols)

    def __getPrediction(self, df, cols, model):
        # get prediction
        prediction = model.predict(df[cols])[0]
        # scale prediction (because of log transformation)
        prediction = np.exp(prediction)
        # round prediction
        prediction = np.round(prediction, -3)
        # make sure prediction is not negative
        prediction = np.max([0, prediction])
        # make sure prediction is an integer
        prediction = int(prediction)
        # return prediction
        return prediction

    def predict(self):
        # checks
        # return None if living_space, type, rooms or zip_code is missing
        if (
            np.isnan(self.living_space)
            or not self.type
            or np.isnan(self.rooms)
            or np.isnan(self.zip_code)
        ):
            return None
        # return None if type is invalid
        if self.type not in types:
            return None

        # generate dataframe
        df = self.__generateDataFrame(cols, data_plz)
        # scale dataframe
        df = self.__scaleDataFrame(df, cols, scaler)
        # return prediction
        return self.__getPrediction(df, cols, model)
