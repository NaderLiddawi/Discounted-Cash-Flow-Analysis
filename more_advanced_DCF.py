import logging
from decimal import Decimal, getcontext, DivisionByZero, InvalidOperation
from statistics import median
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import re
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
getcontext().prec = 6  # Set decimal precision

# ======================
# Configuration Settings
# ======================
DEFAULT_TAX_RATE = Decimal('0.21')
MIN_GROWTH_RATE = Decimal('-0.20')
MAX_GROWTH_RATE = Decimal('0.20')
DEFAULT_MARKET_RISK_PREMIUM = Decimal('0.05')
DEFAULT_RISK_FREE_RATE = Decimal('0.03')
DEFAULT_COST_OF_DEBT = Decimal('0.05')
DEFAULT_WACC = Decimal('0.08')

# ======================
# Utility Functions
# ======================
def to_decimal(value) -> Decimal:
    """
    Safely convert a value to Decimal.
    If conversion fails, log the error and return Decimal(0).
    """
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as e:
        logging.error(f"Conversion error: {e}")
        return Decimal('0')

# ======================
# Custom Exception
# ======================
class FinancialDataError(Exception):
    """Custom exception for errors in financial data processing."""
    pass

# ======================
# Data Fetching and Normalization
# ======================
class FinancialDataFetcher:
    """
    Handles fetching and normalizing financial data from Yahoo Finance.
    Splitting the data retrieval from business logic allows for easier testing
    and maintenance.
    """
    def __init__(self, ticker_symbol: str):
        self.ticker_symbol = ticker_symbol
        self.ticker = None
        self.cashflow = None
        self.income_statement = None
        self.balance_sheet = None

    def fetch_data(self) -> None:
        """Fetch and normalize financial statements."""
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
            logging.error(f"Error fetching data for {self.ticker_symbol}: {e}")
            raise FinancialDataError(f"Failed to fetch financial data: {e}")

    @staticmethod
    def _normalize_index(index: pd.Index) -> pd.Index:
        """
        Normalize line item names to lower-case alphanumeric strings.
        This step ensures that subsequent matching is more flexible.
        """
        normalized = index.str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True)
        normalized = normalized.str.replace(r'\s+', ' ', regex=True).str.strip()
        return pd.Index(normalized)

# ======================
# Line Item Matching
# ======================
class LineItemMatcher:
    """
    Matches financial statement line items using fuzzy matching techniques.
    This class encapsulates the logic needed to resolve discrepancies in naming,
    making the overall valuation process more robust.
    """
    def __init__(self, possible_matches: pd.Index):
        self.possible_matches = possible_matches

    def match(self, line_item: str) -> str:
        """
        Match the provided line item to the best candidate in the list.
        Uses difflib for an initial shortlist, then refines the match using TF-IDF
        and cosine similarity.
        """
        normalized_item = re.sub(r'[^a-z0-9 ]', '', line_item.lower()).strip()
        # Use difflib to get initial close matches
        matches = difflib.get_close_matches(normalized_item, self.possible_matches, n=3, cutoff=0.5)
        if matches:
            items = [normalized_item] + matches
            vectorizer = TfidfVectorizer().fit_transform(items)
            vectors = vectorizer.toarray()
            cosine_similarities = cosine_similarity([vectors[0]], vectors[1:]).flatten()
            max_index = int(np.argmax(cosine_similarities))
            if cosine_similarities[max_index] > 0.5:
                logging.debug(f"Matched '{line_item}' to '{matches[max_index]}' with similarity {cosine_similarities[max_index]}")
                return matches[max_index]
        # Fallback: use cosine similarity against all candidates
        logging.warning(f"Unable to find an exact match for '{line_item}'. Using fallback matching.")
        vectorizer = TfidfVectorizer().fit(self.possible_matches)
        item_vector = vectorizer.transform([normalized_item])
        possible_vectors = vectorizer.transform(self.possible_matches)
        cosine_similarities = cosine_similarity(item_vector, possible_vectors).flatten()
        max_index = int(np.argmax(cosine_similarities))
        if cosine_similarities[max_index] > 0.1:
            logging.debug(f"Matched '{line_item}' to '{self.possible_matches[max_index]}' with similarity {cosine_similarities[max_index]}")
            return self.possible_matches[max_index]
        raise FinancialDataError(f"Unable to find a match for line item: '{line_item}'.")

# ======================
# DCF Valuation Analysis
# ======================
class DCFValuationAnalyzer:
    """
    Performs Discounted Cash Flow (DCF) valuation analysis.
    This class integrates data fetching, normalization, matching, and the actual
    valuation calculations into one cohesive workflow.
    """
    def __init__(self, ticker_symbol: str):
        self.fetcher = FinancialDataFetcher(ticker_symbol)
        self.matcher_cashflow = None
        self.matcher_income = None
        self.matcher_balance = None

    def fetch_financial_data(self) -> None:
        """
        Fetch financial data and initialize line item matchers.
        This separation of data access from business logic increases modularity.
        """
        self.fetcher.fetch_data()
        self.matcher_cashflow = LineItemMatcher(self.fetcher.cashflow.index)
        self.matcher_income = LineItemMatcher(self.fetcher.income_statement.index)
        self.matcher_balance = LineItemMatcher(self.fetcher.balance_sheet.index)

    def calculate_free_cash_flow(self, historical_years: int = 5) -> pd.Series:
        """
        Calculate free cash flow from the cash flow statement.
        It checks if 'free cash flow' is available directly; otherwise, it calculates
        FCF as Operating Cash Flow minus Capital Expenditures.
        """
        cashflow = self.fetcher.cashflow
        try:
            if 'free cash flow' in cashflow.index:
                free_cash_flow = cashflow.loc['free cash flow']
                logging.info("Found 'Free Cash Flow' directly in cash flow statement.")
            else:
                operating_key = self.matcher_cashflow.match('total cash from operating activities')
                capex_key = self.matcher_cashflow.match('capital expenditures')
                operating_cf = cashflow.loc[operating_key]
                capex = cashflow.loc[capex_key]
                free_cash_flow = operating_cf - capex.abs()
                logging.info("Calculated Free Cash Flow as Operating CF minus CapEx.")
            # Limit to available historical years
            available_years = free_cash_flow.shape[1] if isinstance(free_cash_flow, pd.DataFrame) else len(free_cash_flow)
            if available_years < historical_years:
                logging.warning(f"Only {available_years} years available. Using {available_years} years for analysis.")
                historical_years = available_years
            if isinstance(free_cash_flow, pd.DataFrame):
                free_cash_flow = free_cash_flow.iloc[:, :historical_years].squeeze()
            else:
                free_cash_flow = free_cash_flow.sort_index(ascending=True).head(historical_years)
            free_cash_flow = free_cash_flow.apply(to_decimal)
            return free_cash_flow.sort_index(ascending=True)
        except KeyError as e:
            raise FinancialDataError(f"Missing cash flow line item: {e}")
        except Exception as e:
            raise FinancialDataError(f"Error calculating Free Cash Flow: {e}")

    def estimate_growth_rate(self, free_cash_flow: pd.Series) -> Decimal:
        """
        Estimate the growth rate based on historical free cash flow.
        Only periods with positive FCF are used to compute growth, and the median
        is constrained within defined limits.
        """
        try:
            positive_fcf = free_cash_flow[free_cash_flow > 0]
            if len(positive_fcf) < 2:
                logging.warning("Insufficient positive FCF data; defaulting growth rate to 2%.")
                return Decimal('0.02')
            fcf_float = positive_fcf.astype(float)
            growth_rates = fcf_float.pct_change().dropna()
            growth_rates = growth_rates[np.isfinite(growth_rates)]
            if growth_rates.empty:
                logging.warning("No valid growth rates found; defaulting to 2% growth.")
                return Decimal('0.02')
            median_growth = Decimal(str(median(growth_rates)))
            constrained_growth = max(MIN_GROWTH_RATE, min(median_growth, MAX_GROWTH_RATE))
            logging.info(f"Estimated growth rate: {constrained_growth}")
            return constrained_growth
        except Exception as e:
            logging.error(f"Error estimating growth rate: {e}")
            return Decimal('0.02')

    def _find_interest_expense(self) -> Optional[Decimal]:
        """
        Retrieve interest expense from the income statement.
        Uses the matching logic to locate the appropriate line item.
        """
        income = self.fetcher.income_statement
        try:
            interest_key = self.matcher_income.match('interest expense')
            interest_expense = to_decimal(income.loc[interest_key].sum())
            logging.info(f"Interest Expense: {interest_expense}")
            return abs(interest_expense) if interest_expense else None
        except FinancialDataError:
            logging.warning("Interest expense not found in income statement.")
            return None
        except Exception as e:
            logging.error(f"Error retrieving interest expense: {e}")
            return None

    def _estimate_tax_rate(self) -> Decimal:
        """
        Estimate the effective tax rate from the income statement.
        Falls back on a default if required line items are missing.
        """
        income = self.fetcher.income_statement
        try:
            income_before_tax_key = self.matcher_income.match('income before tax')
            tax_key = self.matcher_income.match('income tax expense')
            income_before_tax = to_decimal(income.loc[income_before_tax_key].sum())
            tax_expense = to_decimal(income.loc[tax_key].sum())
            if income_before_tax != 0:
                tax_rate = abs(tax_expense / income_before_tax)
                logging.info(f"Estimated Tax Rate: {tax_rate}")
                return tax_rate
            else:
                logging.warning("Income before tax is zero; using default tax rate.")
        except FinancialDataError:
            logging.warning("Tax line items missing; using default tax rate.")
        except Exception as e:
            logging.error(f"Error estimating tax rate: {e}")
        return DEFAULT_TAX_RATE

    def _calculate_capital_structure(self) -> Tuple[Decimal, Decimal]:
        """
        Calculate the market values of equity and debt from the balance sheet.
        If explicit equity is missing, it calculates equity as Total Assets minus Total Liabilities.
        """
        balance = self.fetcher.balance_sheet
        try:
            equity_keys = ['total shareholders equity', 'total stockholders equity', 'shareholders equity']
            equity_value = None
            for key in equity_keys:
                try:
                    matched_key = self.matcher_balance.match(key)
                    equity_value = to_decimal(balance.loc[matched_key].iloc[0])
                    logging.info(f"Equity found using key '{matched_key}': {equity_value}")
                    break
                except FinancialDataError:
                    continue
            if equity_value is None:
                logging.warning("Equity not found; calculating as Assets minus Liabilities.")
                total_assets = to_decimal(balance.loc[self.matcher_balance.match('total assets')].iloc[0])
                total_liabilities = to_decimal(balance.loc[self.matcher_balance.match('total liabilities')].iloc[0])
                equity_value = total_assets - total_liabilities
                logging.info(f"Calculated Equity: {equity_value}")
            debt_keys = ['total debt', 'long term debt', 'short term debt']
            total_debt = Decimal('0')
            for key in debt_keys:
                try:
                    matched_key = self.matcher_balance.match(key)
                    debt_amount = to_decimal(balance.loc[matched_key].iloc[0])
                    total_debt += debt_amount
                    logging.info(f"Debt found using key '{matched_key}': {debt_amount}")
                except FinancialDataError:
                    continue
            if total_debt == 0:
                logging.warning("No debt found; defaulting to 0.")
            return equity_value, total_debt
        except FinancialDataError as e:
            raise FinancialDataError(f"Error calculating capital structure: {e}")
        except Exception as e:
            raise FinancialDataError(f"Unexpected error in capital structure: {e}")

    def _get_current_stock_price(self) -> Decimal:
        """
        Retrieve the current stock price from ticker history.
        Ensures that the latest available 'Close' price is used.
        """
        try:
            price = to_decimal(self.fetcher.ticker.history(period='1d')['Close'].iloc[-1])
            logging.info(f"Current stock price: {price}")
            return price
        except Exception as e:
            raise FinancialDataError(f"Error retrieving current stock price: {e}")

    def _calculate_wacc(self, equity: Decimal, debt: Decimal, tax_rate: Decimal) -> Decimal:
        """
        Calculate the Weighted Average Cost of Capital (WACC).
        Combines the cost of equity (via CAPM) and the cost of debt, weighted by their market values.
        """
        try:
            beta = to_decimal(self.fetcher.ticker.info.get('beta', 1.0))
            cost_of_equity = DEFAULT_RISK_FREE_RATE + beta * DEFAULT_MARKET_RISK_PREMIUM
            logging.info(f"Cost of Equity: {cost_of_equity}")
            interest_expense = self._find_interest_expense()
            if interest_expense and debt != 0:
                try:
                    avg_interest = (interest_expense / debt).quantize(Decimal('0.0001'))
                except DivisionByZero:
                    avg_interest = DEFAULT_COST_OF_DEBT
                    logging.warning("Division by zero in debt calculation; using default cost of debt.")
            else:
                avg_interest = DEFAULT_COST_OF_DEBT
                logging.warning("Interest expense not available; using default cost of debt.")
            cost_of_debt = avg_interest
            logging.info(f"Cost of Debt: {cost_of_debt}")
            total_capital = equity + debt
            if total_capital == 0:
                raise FinancialDataError("Total capital is zero; cannot calculate WACC.")
            equity_weight = equity / total_capital
            debt_weight = debt / total_capital
            wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (Decimal('1') - tax_rate))
            wacc = wacc.quantize(Decimal('0.0001'))
            logging.info(f"Calculated WACC: {wacc}")
            return wacc
        except Exception as e:
            logging.error(f"Error calculating WACC: {e}")
            return DEFAULT_WACC

    def _project_free_cash_flow(self, fcf_series: pd.Series, growth_rate: Decimal, years: int) -> pd.Series:
        """
        Project future free cash flow for a specified number of years.
        The projection uses the most recent available FCF and compounds it at the estimated growth rate.
        """
        latest_fcf = fcf_series.iloc[-1]  # Use the most recent year's value
        projected = [latest_fcf * ((Decimal('1') + growth_rate) ** i) for i in range(1, years + 1)]
        # Create a date index for the projections by adding years to the last date
        last_date = pd.to_datetime(fcf_series.index[-1])
        projected_dates = [last_date + pd.DateOffset(years=i) for i in range(1, years + 1)]
        proj_series = pd.Series(projected, index=projected_dates)
        logging.info(f"Projected FCF for {years} years.")
        return proj_series

    def _calculate_terminal_value(self, projected_fcf: pd.Series, wacc: Decimal,
                                  terminal_growth_rate: Decimal, projection_years: int) -> Decimal:
        """
        Calculate the terminal value using the perpetuity growth method.
        The terminal value is discounted back to present value using the WACC.
        """
        last_fcf = projected_fcf.iloc[-1]
        if wacc <= terminal_growth_rate:
            raise FinancialDataError("WACC must be greater than terminal growth rate.")
        terminal_value = last_fcf * (Decimal('1') + terminal_growth_rate) / (wacc - terminal_growth_rate)
        discounted_terminal = terminal_value / ((Decimal('1') + wacc) ** projection_years)
        logging.info(f"Terminal Value: {terminal_value}, Discounted Terminal: {discounted_terminal}")
        return discounted_terminal

    def _calculate_intrinsic_value(self, projected_fcf: pd.Series,
                                   terminal_value: Decimal, wacc: Decimal, debt: Decimal) -> Decimal:
        """
        Calculate the intrinsic value per share.
        The enterprise value is computed from the discounted FCF and terminal value,
        from which the net debt is subtracted to find equity value.
        """
        discounted_fcf = [fcf / ((Decimal('1') + wacc) ** i) for i, fcf in enumerate(projected_fcf, start=1)]
        enterprise_value = sum(discounted_fcf) + terminal_value
        equity_value = enterprise_value - debt
        if equity_value < 0:
            logging.warning("Negative equity value calculated; setting intrinsic value to 0.")
            return Decimal('0.00')
        shares_outstanding = to_decimal(self.fetcher.ticker.info.get('sharesOutstanding', 1))
        try:
            intrinsic_value_per_share = (equity_value / shares_outstanding).quantize(Decimal('0.01'))
            logging.info(f"Intrinsic Value per Share: {intrinsic_value_per_share}")
            return intrinsic_value_per_share
        except Exception as e:
            raise FinancialDataError(f"Error calculating intrinsic value per share: {e}")

    def run_dcf_analysis(self, projection_years: int = 5, terminal_growth_rate: float = 0.02,
                         historical_years: int = 5) -> Dict[str, Decimal]:
        """
        Execute the full DCF analysis and return the valuation results.
        The process includes fetching data, estimating growth, calculating WACC, projecting FCF,
        computing terminal value, and finally deriving the intrinsic share price.
        """
        try:
            fcf = self.calculate_free_cash_flow(historical_years=historical_years)
            growth_rate = self.estimate_growth_rate(fcf)
            equity, debt = self._calculate_capital_structure()
            tax_rate = self._estimate_tax_rate()
            wacc = self._calculate_wacc(equity, debt, tax_rate)
            current_price = self._get_current_stock_price()
            projected_fcf = self._project_free_cash_flow(fcf, growth_rate, projection_years)
            terminal_value = self._calculate_terminal_value(projected_fcf, wacc, Decimal(str(terminal_growth_rate)), projection_years)
            intrinsic_value = self._calculate_intrinsic_value(projected_fcf, terminal_value, wacc, debt)
            valuation_status = 'Undervalued' if intrinsic_value > current_price else 'Overvalued'
            return {
                'Intrinsic Value per Share': intrinsic_value,
                'Current Stock Price': current_price,
                'Growth Rate': growth_rate,
                'WACC': wacc,
                'Valuation Status': valuation_status
            }
        except FinancialDataError as e:
            logging.error(f"DCF Analysis Error: {e}")
            return {}
        except Exception as e:
            logging.error(f"Unexpected error in DCF analysis: {e}")
            return {}

# ======================
# User Input Parsing
# ======================
def parse_user_input() -> Tuple[str, int, float, int]:
    """
    Parse user input to extract the ticker symbol, projection years,
    terminal growth rate, and historical years.
    This function isolates input handling from the analysis logic.
    """
    ticker_symbol = input("Enter stock ticker symbol: ").upper().strip()
    if not ticker_symbol:
        raise ValueError("Ticker symbol cannot be empty.")

    try:
        projection_years = int(input("Enter number of projection years (default 5): ").strip() or 5)
        if projection_years <= 0:
            logging.warning("Projection years must be positive. Defaulting to 5.")
            projection_years = 5
    except ValueError:
        logging.warning("Invalid input for projection years. Defaulting to 5.")
        projection_years = 5

    try:
        terminal_growth_rate = float(input("Enter terminal growth rate (as decimal, default 0.02): ").strip() or 0.02)
        if terminal_growth_rate <= 0:
            logging.warning("Terminal growth rate must be positive. Defaulting to 0.02.")
            terminal_growth_rate = 0.02
    except ValueError:
        logging.warning("Invalid input for terminal growth rate. Defaulting to 0.02.")
        terminal_growth_rate = 0.02

    try:
        historical_years = int(input("Enter number of historical years to analyze (default 5): ").strip() or 5)
        if historical_years <= 0:
            logging.warning("Historical years must be positive. Defaulting to 5.")
            historical_years = 5
    except ValueError:
        logging.warning("Invalid input for historical years. Defaulting to 5.")
        historical_years = 5

    return ticker_symbol, projection_years, terminal_growth_rate, historical_years

# ======================
# Main Execution Flow
# ======================
def main():
    try:
        ticker_symbol, projection_years, terminal_growth_rate, historical_years = parse_user_input()
        analyzer = DCFValuationAnalyzer(ticker_symbol)
        analyzer.fetch_financial_data()
        results = analyzer.run_dcf_analysis(
            projection_years=projection_years,
            terminal_growth_rate=terminal_growth_rate,
            historical_years=historical_years
        )
        if results:
            logging.info("\nDiscounted Cash Flow (DCF) Valuation Results:")
            for key, value in results.items():
                logging.info(f"{key}: {value}")
        else:
            logging.error("DCF analysis failed. No results available.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
