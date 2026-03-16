#Online-Retail-II-UCI Market Demand forecasting model

## Run in browser
https://online-retail-demand-forecasting.streamlit.app/



## 1. Overview and business objective
* Project is solving problem of revenue optimization and dynamic pricing for online retail store.
The model's forecasts achieve a 45% better MAE than a naive hypothesis of flat market demand.
Model's forecasts also includes dynamic price optimization which are not only aggressively increasing prices, but predicts how price changes affect on quantity of sales.



## 2. ML approach
* The list of factors why LightGBM is optimal model for this project:

	1) It's great working on large table data, only models problem in this context comparing to XGBoost or CatBoost is worse performance on small datasets,
	   but it's not problem in direct dataset.

	2) LightGBM using histogram-based learning, it helps model to work with 5000+ unique product ids without extreme RAM usage and quality loss.

	3) LightGBM's main feature is leaf-wise trees building, unlike other models that build trees level-wise.
	   Leaf-wise tree is better reducing an error then level-wise, but it's more affected to overfitting.
	   On chaotic retail data overfitting is not such a big problem as huge errors because of noisy data.
	
	4) Only the problem of LightGBM noise sensitivity is solving by setting logarithmic target and reducing noise by bigger forecast sampling.



* Feature engineering explanation:
	
	1) EWMA_Target feature is recency-weighted rolling mean of target's historical data.
 	   Big span of 12 ticks makes EWMA more smoothed and less affected to peaks.
 	   Significant recency-quantity combined feature with 14.8% of total gain.

	2) Price block features needed for second step of model performance, exactly for dynamical pricing.
	   Price features are giving to model information about how price changes may affect on prediction target(sales quantity).
	   Essential for model not only to predict sales, but to find revenue optimization problem solution.
	   Summary price features are near 20% of total gain, what indicates that model is not price-blind.

	3) Lag features are giving to model historical context to particularly understand a sales trend for each product.
	   Summary lags features gain is near 20% of total

        4) Unique customers feature can explain is it demand boom for some product groups or direct product or it's market-wide event related with customers quantity changes
	   Has 5% of total gain


## 3. Tech stack
*Python, pandas, pyarrow, sklearn, LightGBM, Streamlit, FastApi



## 4. Project structure
src/ - holds engine logic, config constants
data/ - holds dataset as a parquet file
frontend/ - holds Streamlit UI
API/ - holds api module for local start
model/ - contains trained model in .joblib format
notebooks/ - contains notebook with model training script and inference notebook to run model without UI


## 5. How to run locally:

	1) Download files from GitHub to local machine:
	bash
	git clone https://github.com/vozniukivan0501-cmyk/Online-Retail-II-UCI.git
   	cd Online-Retail-II-UCI

	2) Install dependencies
	Terminal
	pip install -r frontend/requirements.txt

	3) Run api module
	Terminal
	python -m uvicorn api.API:app --port 8080

	4) Run UI
	Terminal
	streamlit run frontend/st_gui.py

		