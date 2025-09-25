import pandas as pd

SP_df = pd.DataFrame({"strike_price": [24000, 24100, 24200]})
options = [{"label": sp, "value": sp} for sp in SP_df["strike_price"]]
value  = NIFTY_Index.Active_SP if  NIFTY_Index.Active_SP != 0 else None

print(options)