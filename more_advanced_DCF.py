import logging
from decimal import Decimal, getcontext, DivisionByZero
from statistics import median
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import re
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


print("Turn off your VPN before running code")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
getcontext().prec = 6  # Set decimal precision


class FinancialDataError(Exception):
    """Custom exception for financial data retrieval and processing errors."""
    pass


class FinancialDataFetcher:
    """Handles fetching and normalizing financial data from Yahoo Finance."""

    def __init__(self, ticker_symbol: str):
        self.ticker_symbol = ticker_symbol
        self.ticker = None
        self.cashflow = None
        self.income_statement = None
        self.balance_sheet = None

    def fetch_data(self):
        try:
            logging.info(f"Fetching financial data for {self.ticker_symbol}")
            self.ticker = yf.Ticker(self.ticker_symbol)
            self.cashflow = self.ticker.cashflow
            self.income_statement = self.ticker.financials
            self.balance_sheet = self.ticker.balance_sheet

            if self.cashflow.empty or self.income_statement.empty or self.balance_sheet.empty:
                raise FinancialDataError("One or more financial statements are empty.")

            # Normalize indices for flexible lookup
            self.cashflow.index = self._normalize_index(self.cashflow.index)
            self.income_statement.index = self._normalize_index(self.income_statement.index)
            self.balance_sheet.index = self._normalize_index(self.balance_sheet.index)

        except Exception as e:
            print("Cashflow is: ", self.cashflow)
            print("Income Statement is: ", self.income_statement)
            print("Balance sheet is: ", self.balance_sheet)
            raise FinancialDataError(f"Failed to fetch financial data: {e}")

    @staticmethod
    def _normalize_index(index: pd.Index) -> pd.Index:
        """Normalize the line item names in the financial statement index."""
        normalized = index.str.lower()
        normalized = normalized.str.replace(r'[^a-z0-9 ]', '', regex=True)
        normalized = normalized.str.replace(r'\s+', ' ', regex=True).str.strip()
        return pd.Index(normalized)


class LineItemMatcher:
    """Handles matching and predicting financial statement line items."""

    def __init__(self, possible_matches: pd.Index):
        self.possible_matches = possible_matches

    def match(self, line_item: str) -> str:
        normalized_item = re.sub(r'[^a-z0-9 ]', '', line_item.lower()).strip()

        # Initial close matches using difflib
        matches = difflib.get_close_matches(normalized_item, self.possible_matches, n=3, cutoff=0.5)
        if matches:
            # Refine matches using TF-IDF and cosine similarity
            items = [normalized_item] + matches
            vectorizer = TfidfVectorizer().fit_transform(items)
            vectors = vectorizer.toarray()
            cosine_similarities = cosine_similarity([vectors[0]], vectors[1:]).flatten()
            max_index = np.argmax(cosine_similarities)
            if cosine_similarities[max_index] > 0.5:
                logging.debug(
                    f"Matched '{line_item}' to '{matches[max_index]}' with similarity {cosine_similarities[max_index]}")
                return matches[max_index]

        # Fallback to cosine similarity with all possible matches
        logging.warning(
            f"Unable to find an exact match for line item: '{line_item}'. Using cosine similarity for prediction.")
        vectorizer = TfidfVectorizer().fit(self.possible_matches)
        item_vector = vectorizer.transform([normalized_item])
        possible_vectors = vectorizer.transform(self.possible_matches)
        cosine_similarities = cosine_similarity(item_vector, possible_vectors).flatten()
        max_index = np.argmax(cosine_similarities)
        if cosine_similarities[max_index] > 0.1:
            logging.debug(
                f"Matched '{line_item}' to '{self.possible_matches[max_index]}' with similarity {cosine_similarities[max_index]}")
            return self.possible_matches[max_index]

        raise FinancialDataError(f"Unable to find a match for line item: '{line_item}'.")


class DCFValuationAnalyzer:
    """Performs Discounted Cash Flow (DCF) valuation analysis."""

    DEFAULT_TAX_RATE = Decimal('0.21')  # Default corporate tax rate
    MIN_GROWTH_RATE = Decimal('-0.20')  # Minimum growth rate for projections (e.g., -20%)
    MAX_GROWTH_RATE = Decimal('0.20')  # Maximum growth rate for projections
    DEFAULT_MARKET_RISK_PREMIUM = Decimal('0.05')  # 5% default market risk premium

    def __init__(self, ticker_symbol: str):
        self.fetcher = FinancialDataFetcher(ticker_symbol)
        self.matcher_cashflow = None
        self.matcher_income = None
        self.matcher_balance = None

    def fetch_financial_data(self):
        self.fetcher.fetch_data()
        # Initialize matchers for each financial statement
        self.matcher_cashflow = LineItemMatcher(self.fetcher.cashflow.index)
        self.matcher_income = LineItemMatcher(self.fetcher.income_statement.index)
        self.matcher_balance = LineItemMatcher(self.fetcher.balance_sheet.index)

    def calculate_free_cash_flow(self, historical_years: int = 5) -> pd.Series:
        """Calculate Free Cash Flow from the cash flow statement, limited to a specified number of historical years."""
        cashflow = self.fetcher.cashflow
        try:
            if 'free cash flow' in cashflow.index:
                free_cash_flow = cashflow.loc['free cash flow']
                logging.info("Found 'Free Cash Flow' directly in cash flow statement.")
            else:
                # Calculate: Operating Cash Flow - Capital Expenditures
                operating_cf_key = self.matcher_cashflow.match('total cash from operating activities')
                capex_key = self.matcher_cashflow.match('capital expenditures')
                operating_cash_flow = cashflow.loc[operating_cf_key]
                capital_expenditures = cashflow.loc[capex_key]
                free_cash_flow = operating_cash_flow - capital_expenditures.abs()
                logging.info("Calculated 'Free Cash Flow' as Operating Cash Flow minus Capital Expenditures.")

            # Limit to the specified number of historical years
            available_years = free_cash_flow.shape[1] if isinstance(free_cash_flow, pd.DataFrame) else len(
                free_cash_flow)
            if available_years < historical_years:
                logging.warning(
                    f"Requested {historical_years} historical years, but only {available_years} available. Using {available_years} years.")
                historical_years = available_years

            if isinstance(free_cash_flow, pd.DataFrame):
                free_cash_flow = free_cash_flow.iloc[:, :historical_years].squeeze()
            else:
                free_cash_flow = free_cash_flow.sort_index(ascending=True).head(historical_years)

            free_cash_flow = free_cash_flow.apply(lambda x: Decimal(x))
            return free_cash_flow.sort_index(ascending=True)
        except KeyError as e:
            raise FinancialDataError(f"Missing required cash flow line item: {e}")
        except Exception as e:
            raise FinancialDataError(f"Error calculating Free Cash Flow: {e}")

    def estimate_growth_rate(self, free_cash_flow: pd.Series) -> Decimal:
        """Estimate growth rate based on historical Free Cash Flow."""
        try:
            # Only consider periods with positive FCF for growth rate estimation
            positive_fcf = free_cash_flow[free_cash_flow > 0]
            if len(positive_fcf) < 2:
                logging.warning("Not enough positive FCF data to estimate growth rate. Using default 2% growth.")
                return Decimal('0.02')  # Default growth rate

            fcf = positive_fcf.astype(float)
            growth_rates = fcf.pct_change().dropna()
            growth_rates = growth_rates[np.isfinite(growth_rates)]
            print(f"****GROWTH RATES: {growth_rates}")

            if growth_rates.empty:
                logging.warning("No valid historical growth rates found from positive FCF. Using default 2% growth.")
                return Decimal('0.02')  # Default growth rate

            median_growth = Decimal(str(median(growth_rates)))
            constrained_growth = max(self.MIN_GROWTH_RATE, min(median_growth, self.MAX_GROWTH_RATE))
            logging.info(f"Estimated growth rate: {constrained_growth}")
            return constrained_growth
        except Exception as e:
            logging.error(f"Error estimating growth rate: {e}")
            return Decimal('0.02')  # Default growth rate

    def _find_interest_expense(self) -> Optional[Decimal]:
        """Find interest expense from the income statement."""
        income = self.fetcher.income_statement
        try:
            interest_expense_key = self.matcher_income.match('interest expense')
            interest_expense = Decimal(income.loc[interest_expense_key].sum())
            logging.info(f"Found Interest Expense: {interest_expense}")
            return abs(interest_expense) if interest_expense else None
        except FinancialDataError:
            logging.warning("Interest expense line item not found.")
            return None
        except Exception as e:
            logging.error(f"Error retrieving interest expense: {e}")
            return None

    def _estimate_tax_rate(self) -> Decimal:
        """Estimate tax rate from the income statement."""
        income = self.fetcher.income_statement
        try:
            income_before_tax_key = self.matcher_income.match('income before tax')
            tax_expense_key = self.matcher_income.match('income tax expense')

            income_before_tax = Decimal(income.loc[income_before_tax_key].sum())
            tax_expense = Decimal(income.loc[tax_expense_key].sum())

            if income_before_tax != 0:
                tax_rate = abs(tax_expense / income_before_tax)
                logging.info(f"Estimated Tax Rate: {tax_rate}")
                return tax_rate
            else:
                logging.warning("Income before tax is zero. Using default tax rate.")
        except FinancialDataError:
            logging.warning("Tax-related line items not found. Using default tax rate.")
        except Exception as e:
            logging.error(f"Error estimating tax rate: {e}")

        return self.DEFAULT_TAX_RATE

    def _calculate_capital_structure(self) -> Tuple[Decimal, Decimal]:
        """Calculate the market values of equity and debt."""
        balance_sheet = self.fetcher.balance_sheet
        try:
            # Equity
            equity_keys = [
                'total shareholders equity',
                'total stockholders equity',
                'shareholders equity'
            ]
            equity_value = None
            for key in equity_keys:
                try:
                    equity_value_key = self.matcher_balance.match(key)
                    equity_value = Decimal(balance_sheet.loc[equity_value_key].iloc[0])
                    logging.info(f"Found Equity Value using '{equity_value_key}': {equity_value}")
                    break
                except FinancialDataError:
                    continue

            if equity_value is None:
                # Calculate Equity = Assets - Liabilities
                logging.warning("Shareholders' equity not found. Calculating as Total Assets - Total Liabilities.")
                total_assets_key = self.matcher_balance.match('total assets')
                total_liabilities_key = self.matcher_balance.match('total liabilities')
                total_assets = Decimal(balance_sheet.loc[total_assets_key].iloc[0])
                total_liabilities = Decimal(balance_sheet.loc[total_liabilities_key].iloc[0])
                equity_value = total_assets - total_liabilities
                logging.info(f"Calculated Equity Value: {equity_value}")

            # Debt
            debt_keys = [
                'total debt',
                'long term debt',
                'short long term debt',
                'short term debt'
            ]
            total_debt = Decimal('0')
            for key in debt_keys:
                try:
                    debt_key = self.matcher_balance.match(key)
                    debt_amount = Decimal(balance_sheet.loc[debt_key].iloc[0])
                    total_debt += debt_amount
                    logging.info(f"Found Debt Value using '{debt_key}': {debt_amount}")
                except FinancialDataError:
                    continue

            if total_debt == 0:
                logging.warning("Total debt not found. Assuming total debt is zero.")
            else:
                logging.info(f"Total Debt: {total_debt}")

            return equity_value, total_debt
        except FinancialDataError as e:
            raise FinancialDataError(f"Error calculating capital structure: {e}")
        except Exception as e:
            raise FinancialDataError(f"Unexpected error in capital structure calculation: {e}")

    def _get_current_stock_price(self) -> Decimal:
        """Retrieve the current stock price."""
        try:
            price = Decimal(self.fetcher.ticker.history(period='1d')['Close'].iloc[-1])
            logging.info(f"Current Stock Price: {price}")
            return price
        except Exception as e:
            raise FinancialDataError(f"Error retrieving current stock price: {e}")

    def _calculate_wacc(self, equity: Decimal, debt: Decimal, tax_rate: Decimal) -> Decimal:
        """
        Calculate the Weighted Average Cost of Capital (WACC).

        Args:
            equity (Decimal): Market value of equity
            debt (Decimal): Market value of debt
            tax_rate (Decimal): Corporate tax rate

        Returns:
            Decimal: WACC
        """
        try:
            # Calculate Cost of Equity using CAPM
            beta = Decimal(str(self.fetcher.ticker.info.get('beta', 1.0)))
            risk_free_rate = Decimal('0.03')  # Assume 3% risk-free rate (can be made configurable)
            market_risk_premium = self.DEFAULT_MARKET_RISK_PREMIUM  # 5% default

            cost_of_equity = risk_free_rate + beta * market_risk_premium
            logging.info(f"Calculated Cost of Equity (CAPM): {cost_of_equity}")

            # Cost of Debt
            interest_expense = self._find_interest_expense()
            if interest_expense and debt != 0:
                # Estimate average interest rate on debt
                try:
                    average_interest_rate = (interest_expense / debt).quantize(Decimal('0.0001'))
                except DivisionByZero:
                    average_interest_rate = Decimal('0.05')  # Default 5%
                    logging.warning("Debt is zero during Cost of Debt calculation. Using default 5%.")
                logging.info(f"Estimated Cost of Debt: {average_interest_rate}")
            else:
                average_interest_rate = Decimal('0.05')  # Default 5%
                logging.warning("Using default Cost of Debt: 5%")

            cost_of_debt = average_interest_rate
            logging.info(f"Cost of Debt: {cost_of_debt}")

            total_capital = equity + debt
            if total_capital == 0:
                raise FinancialDataError("Total capital is zero. Cannot calculate WACC.")

            equity_weight = equity / total_capital
            debt_weight = debt / total_capital

            wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (Decimal('1') - tax_rate))
            wacc = wacc.quantize(Decimal('0.0001'))
            logging.info(f"Calculated WACC: {wacc}")
            return wacc
        except Exception as e:
            logging.error(f"Error calculating WACC: {e}")
            return Decimal('0.08')  # Default WACC

    def run_dcf_analysis(self, projection_years: int = 5, terminal_growth_rate: float = 0.02,
                         historical_years: int = 5) -> Dict[str, Decimal]:
        """Run comprehensive DCF valuation analysis."""
        try:
            free_cash_flow = self.calculate_free_cash_flow(historical_years=historical_years)

            # Print historical FCF
            logging.info("\nHistorical Free Cash Flow:")
            fcf_df = free_cash_flow.reset_index()
            fcf_df.columns = ['Year', 'Free Cash Flow']
            # Format FCF with commas and two decimal places
            fcf_df['Free Cash Flow'] = fcf_df['Free Cash Flow'].apply(lambda x: f"{x:,.2f}")
            logging.info(fcf_df.to_string(index=False))

            growth_rate = self.estimate_growth_rate(free_cash_flow)
            equity, debt = self._calculate_capital_structure()
            tax_rate = self._estimate_tax_rate()
            wacc = self._calculate_wacc(equity, debt, tax_rate)
            current_price = self._get_current_stock_price()

            projected_fcf = self._project_free_cash_flow(free_cash_flow, growth_rate, projection_years)
            terminal_value = self._calculate_terminal_value(projected_fcf, wacc, Decimal(str(terminal_growth_rate)),
                                                            projection_years)
            intrinsic_value = self._calculate_intrinsic_value(projected_fcf, terminal_value, wacc, debt)

            return {
                'Intrinsic Value per Share': intrinsic_value,
                'Current Stock Price': current_price,
                'Growth Rate': growth_rate,
                'WACC': wacc,
                'Valuation Status': 'Undervalued' if intrinsic_value > current_price else 'Overvalued'
            }
        except FinancialDataError as e:
            logging.error(f"DCF Analysis Error: {e}")
            return {}
        except Exception as e:
            logging.error(f"Unexpected error during DCF analysis: {e}")
            return {}

    def _project_free_cash_flow(self, fcf_series: pd.Series, growth_rate: Decimal, years: int) -> pd.Series:
        """Project future Free Cash Flow based on historical data."""
        latest_fcf = fcf_series.iloc[0]
        projected_fcf = [latest_fcf * ((Decimal('1') + growth_rate) ** i) for i in range(1, years + 1)]
        projected_years = [fcf_series.index[0] + pd.DateOffset(years=i) for i in range(1, years + 1)]
        projected_series = pd.Series(projected_fcf, index=projected_years)
        logging.info(f"Projected Free Cash Flow for next {years} years.")
        return projected_series

    def _calculate_terminal_value(self, projected_fcf: pd.Series, wacc: Decimal,
                                  terminal_growth_rate: Decimal, projection_years: int) -> Decimal:
        """Calculate discounted terminal value."""
        last_fcf = projected_fcf.iloc[-1]
        if wacc <= terminal_growth_rate:
            raise FinancialDataError("WACC must be greater than terminal growth rate for terminal value calculation.")
        terminal_value = last_fcf * (Decimal('1') + terminal_growth_rate) / (wacc - terminal_growth_rate)
        discounted_terminal = terminal_value / ((Decimal('1') + wacc) ** projection_years)
        logging.info(f"Terminal Value: {terminal_value}, Discounted Terminal Value: {discounted_terminal}")
        return discounted_terminal

    def _calculate_intrinsic_value(self, projected_fcf: pd.Series,
                                   terminal_value: Decimal, wacc: Decimal, debt: Decimal) -> Decimal:
        """Calculate intrinsic value per share."""
        discounted_fcf = []
        for i, fcf in enumerate(projected_fcf, start=1):
            discounted_value = fcf / ((Decimal('1') + wacc) ** i)
            discounted_fcf.append(discounted_value)
        total_fcf = sum(discounted_fcf)
        enterprise_value = total_fcf + terminal_value
        equity_value = enterprise_value - debt

        if equity_value < 0:
            logging.warning("Calculated Equity Value is negative. Setting Intrinsic Value per Share to zero.")
            return Decimal('0.00')

        shares_outstanding = self.fetcher.ticker.info.get('sharesOutstanding', 1)
        try:
            shares_outstanding = Decimal(shares_outstanding)
            intrinsic_value_per_share = equity_value / shares_outstanding
            intrinsic_value_per_share = intrinsic_value_per_share.quantize(Decimal('0.01'))
            logging.info(f"Intrinsic Value per Share: {intrinsic_value_per_share}")
            return intrinsic_value_per_share
        except Exception as e:
            raise FinancialDataError(f"Error calculating intrinsic value per share: {e}")


def main():
    try:
        ticker_symbol = input("Enter stock ticker symbol: ").upper().strip()
        if not ticker_symbol:
            logging.error("Ticker symbol cannot be empty.")
            return

        # Get number of projection years
        projection_years_input = input("Enter number of projection years (default 5): ").strip()
        if projection_years_input:
            try:
                projection_years = int(projection_years_input)
                if projection_years <= 0:
                    logging.warning("Projection years must be positive. Using default value of 5.")
                    projection_years = 5
            except ValueError:
                logging.warning("Invalid input for projection years. Using default value of 5.")
                projection_years = 5
        else:
            projection_years = 5

        # Get terminal growth rate
        terminal_growth_input = input("Enter terminal growth rate (as decimal, default 0.02): ").strip()
        if terminal_growth_input:
            try:
                terminal_growth_rate = float(terminal_growth_input)
                if terminal_growth_rate <= 0:
                    logging.warning("Terminal growth rate must be positive. Using default value of 0.02.")
                    terminal_growth_rate = 0.02
            except ValueError:
                logging.warning("Invalid input for terminal growth rate. Using default value of 0.02.")
                terminal_growth_rate = 0.02
        else:
            terminal_growth_rate = 0.02

        # Get number of historical years to analyze
        historical_years_input = input("Enter number of historical years to analyze (default 5): ").strip()
        if historical_years_input:
            try:
                historical_years = int(historical_years_input)
                if historical_years <= 0:
                    logging.warning("Historical years must be positive. Using default value of 5.")
                    historical_years = 5
            except ValueError:
                logging.warning("Invalid input for historical years. Using default value of 5.")
                historical_years = 5
        else:
            historical_years = 5

        analyzer = DCFValuationAnalyzer(ticker_symbol)
        analyzer.fetch_financial_data()
        results = analyzer.run_dcf_analysis(projection_years=projection_years,
                                            terminal_growth_rate=terminal_growth_rate,
                                            historical_years=historical_years)

        if results:
            logging.info("\nDiscounted Cash Flow (DCF) Valuation Results:")
            for key, value in results.items():
                logging.info(f"{key}: {value}")
        else:
            logging.error("DCF analysis failed. No results to display.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
