The Discounted Cash Flow (DCF) Valuation Tool is designed to estimate a stock's intrinsic value and assess whether it is under- or overvalued compared to its current market price. It begins by retrieving financial statements (cash flow, income statement, and balance sheet) for a specified stock ticker using Yahoo Finance's API (yfinance). To ensure flexible lookup, the tool normalizes financial statement indices. It employs advanced text similarity techniques, such as TF-IDF and cosine similarity, to accurately match line items from the financial statements, even with naming variations.

The tool calculates Free Cash Flow (FCF) either directly or by deriving it as operating cash flow minus capital expenditures. Historical FCF data is analyzed to estimate growth rates, which are constrained within predefined limits. Additional key metrics, such as corporate tax rate and interest expense, are extracted to support valuation. The tool evaluates the company’s capital structure, computing the market values of equity and debt, and uses the Weighted Average Cost of Capital (WACC)—calculated via the Capital Asset Pricing Model (CAPM) for equity and the cost of debt—to discount future cash flows. It then projects FCF over user-defined years, calculates the terminal value using a perpetual growth model, and derives the intrinsic value per share by discounting these values back to the present.

User inputs such as stock ticker, projection years, terminal growth rate, and historical analysis years are accepted, with default options provided for flexibility. The tool includes robust error handling for missing data, invalid inputs, or calculation issues and provides detailed logging for tracking and debugging. With its ability to deliver key metrics like intrinsic value, growth rate, and WACC, this tool is ideal for financial analysts or developers aiming to make informed investment decisions.






