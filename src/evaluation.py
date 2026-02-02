import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

class ModelEvaluator:
    def __init__(self, df_full, df_incomplete, metadata):
        self.df_full = df_full
        self.df_incomplete = df_incomplete
        self.metadata = metadata
        self.mask = df_incomplete.isnull()

    def calculate_rmse(self, df_pred):
        """Calcul du RMSE global sur les données manquantes uniquement"""
        y_true = self.df_full.values[self.mask.values]
        y_pred = df_pred.values[self.mask.values]
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def error_by_sector(self, df_pred):
        """Calcule l'erreur moyenne pour chaque secteur industriel"""
        sector_errors = {}
        sectors = self.metadata['sector'].unique()
        
        for sector in sectors:
            # On récupère les IDs des obligations du secteur
            bonds = self.metadata[self.metadata['sector'] == sector].index
            
            # On isole les données pour ce secteur
            y_true = self.df_full.loc[bonds].values[self.mask.loc[bonds].values]
            y_pred = df_pred.loc[bonds].values[self.mask.loc[bonds].values]
            
            if len(y_true) > 0:
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                sector_errors[sector] = rmse
                
        return pd.Series(sector_errors).sort_values()

    def error_by_rating(self, df_pred):
        """Calcule l'erreur moyenne par note de crédit (Rating)"""
        rating_errors = {}
        ratings = self.metadata['rating'].unique()
        
        for rat in ratings:
            bonds = self.metadata[self.metadata['rating'] == rat].index
            y_true = self.df_full.loc[bonds].values[self.mask.loc[bonds].values]
            y_pred = df_pred.loc[bonds].values[self.mask.loc[bonds].values]
            
            if len(y_true) > 0:
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                rating_errors[rat] = rmse
                
        return pd.Series(rating_errors).sort_values()