# Numeric features → StandardScaler → KMeans / PCA / DBSCAN 
# Categorical features → OneHotEncoder → KMeans / PCA / DBSCAN

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureEngineer:

    def __init__(self, df: pd.DataFrame):
        print("  FeatureEngineer Initialized")
        print(f"  Input shape : {df.shape[0]:,} rows × {df.shape[1]} columns")

        self.df            = df.copy()
        self.pipeline      = None
        self.feature_names_ = None
        self.target_encoders = {}
    
    # 
    def datetime_features(self):
        print("Create the Date feature")

        # Ensure Date column is datetime
        self.df['date'] =  pd.to_datetime(self.df["date"], errors="coerce")

        # create a hour of day (0-23) from datetime

        self.df["hour"] = self.df["date"].dt.hour

        # create Day name (Monday-Sunday)

        self.df["day_name"] = self.df["date"].dt.day_name()

        # create a Month
        self.df["month"] = self.df["date"].dt.month

        # create a Season
        def get_season(month):

            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            else:
                return "Autumn"

        self.df["season"] = self.df["month"].apply(get_season)

        # create the  Boolean flag for weekend crimes

        self.df["Weekend_crimes"] = self.df["day_name"].isin(["Saturday", "Sunday"]).astype(int)


        # create Crime_Severity_Score Numerical score based on crime type
    def crime_severity(self):

        print("Creating crime severity score")

        crime_severity = {
            "HOMICIDE": 5,
            "KIDNAPPING": 5,
            "CRIMINAL SEXUAL ASSAULT": 5,
            "CRIM SEXUAL ASSAULT": 5,
            "HUMAN TRAFFICKING": 5,

            "ROBBERY": 4,
            "ASSAULT": 4,
            "BURGLARY": 4,
            "ARSON": 4,
            "WEAPONS VIOLATION": 4,

            "BATTERY": 3,
            "MOTOR VEHICLE THEFT": 3,
            "CRIMINAL DAMAGE": 3,
            "STALKING": 3,
            "INTIMIDATION": 3,

            "THEFT": 2,
            "CRIMINAL TRESPASS": 2,
            "NARCOTICS": 2,
            "SEX OFFENSE": 2,
            "PROSTITUTION": 2,
            "GAMBLING": 2,

            "PUBLIC PEACE VIOLATION": 1,
            "LIQUOR LAW VIOLATION": 1,
            "PUBLIC INDECENCY": 1,
            "OBSCENITY": 1,
            "OTHER OFFENSE": 1,
            "OTHER NARCOTIC VIOLATION": 1,
            "NON-CRIMINAL": 1,
            "RITUALISM": 1
            }
        
        self.df["Crime_Severity_Score"] = self.df["primary_type"].map(crime_severity)
        self.df["Crime_Severity_Score"] = self.df["Crime_Severity_Score"].fillna(1)

        print("Crime severity feature created")


        # create a Geographic Feature Engineering
    def geographic_features(self):

        print("Creating geographic features")
            #  Latitude/Longitude binning (divide city into grid)

        self.df["lat_bin"] = pd.cut(self.df["latitude"], bins=10, labels=False).fillna(-1)
        self.df["lon_bin"] = pd.cut(self.df["longitude"], bins=10, labels=False).fillna(-1)

            # District feature
        self.df["district_cluster"] = self.df["district"]

        print("Geographic features created")

    # Encoding 
    
    def encode_features(self):

        print("Encoding categorical features")

        categorical_cols = ["season", "primary_type", "day_name"]

        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        encoded = self.encoder.fit_transform(self.df[categorical_cols])

        encoded_df = pd.DataFrame(
            encoded,
            columns=self.encoder.get_feature_names_out(categorical_cols)
        )

        # # Drop original categorical columns
        # self.df = self.df.drop(columns=categorical_cols)

        # Concatenate encoded features
        self.df = pd.concat([self.df.reset_index(drop=True), encoded_df], axis=1)



    # SCALING


    def scale_features(self):

        print("Scaling numeric features")

        numeric_cols = [
            "hour", "month", "Weekend_crimes",
            "Crime_Severity_Score",
            "latitude", "longitude",
            "lat_bin", "lon_bin", "district_cluster"
        ]

        self.scaler = StandardScaler()

        self.df[numeric_cols] = self.scaler.fit_transform(self.df[numeric_cols])


    # FINAL TRANSFORM


    def transform(self):

        return self.df


# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":

    input_path = r"C:\Users\SAKTHI\Desktop\myproject\PatrolIQ\data\Cleaned_chiago_data.csv"
    output_dir = r"C:\Users\SAKTHI\Desktop\myproject\PatrolIQ\data\feature_data"

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)

    fe = FeatureEngineer(df)

    fe.datetime_features()
    fe.crime_severity()
    fe.geographic_features()
    fe.encode_features()
    fe.scale_features()

    X = fe.transform()

    # Save dataset
    feature_path = os.path.join(output_dir, "feature_dataset.csv")
    X.to_csv(feature_path, index=False)

    print("Saved:", feature_path)

    # Save encoder & scaler
    joblib.dump(fe.encoder, os.path.join(output_dir, "encoder.pkl"))
    joblib.dump(fe.scaler, os.path.join(output_dir, "scaler.pkl"))

    print("Feature engineering complete!")