import pandas as pd

class TradingStrategy:
    def __init__(self, threshold=15):
        """
        threshold : l'écart minimum (en bps) pour considérer qu'une obligation 
        est vraiment "pas chère" ou "trop chère".
        """
        self.threshold = threshold

    def generate_signals(self, df_market, df_fair_value):
        """
        Identifie les opportunités Rich/Cheap.
        Signal = OAS Marché - OAS Fair Value
        """
        # On calcule l'écart (le "Z-score" simplifié)
        spread_diff = df_market - df_fair_value
        
        signals = pd.DataFrame(index=df_market.index, columns=df_market.columns)
        
        # Si Spread Marché est bcp plus haut que Fair Value -> CHEAP (Achat)
        signals[spread_diff > self.threshold] = "CHEAP (BUY)"
        
        # Si Spread Marché est bcp plus bas que Fair Value -> RICH (SELL)
        signals[spread_diff < -self.threshold] = "RICH (SELL)"
        
        # Sinon -> NEUTRAL
        signals = signals.fillna("NEUTRAL")
        
        return signals, spread_diff

    def get_top_opportunities(self, spread_diff, date):
        """Retourne les 5 meilleures opportunités à une date donnée"""
        daily_diff = spread_diff[date].sort_values(ascending=False)
        
        top_buy = daily_diff.head(5) # Plus gros écart positif
        top_sell = daily_diff.tail(5) # Plus gros écart négatif
        
        return top_buy, top_sell