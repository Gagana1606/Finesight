from pathlib import Path

from streamlit.testing.v1 import AppTest


# ---------------------------------------------------------
# APP PATH
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
APP_PATH = BASE_DIR / "app" / "app.py"


# ---------------------------------------------------------
# HELPER
# ---------------------------------------------------------

def run_app(ticker="", period="1y"):
    app = AppTest.from_file(str(APP_PATH))

    app.run(timeout=60)

    # Set ticker if supplied
    if ticker is not None:
        text_inputs = app.text_input

        if len(text_inputs) > 0:
            text_inputs[0].set_value(ticker)

    # Set period if supplied
    if period is not None:
        selectboxes = app.selectbox

        if len(selectboxes) > 0:
            # Find the Analysis Period selectbox
            period_box = None

            for box in selectboxes:
                if getattr(box, "label", "") == "Analysis Period":
                    period_box = box
                    break

            if period_box is None:
                period_box = selectboxes[0]

            period_box.set_value(period)

    # Run again after changing inputs
    app.run(timeout=60)

    return app


# ---------------------------------------------------------
# SAFE EXCEPTION CHECK
# ---------------------------------------------------------

def assert_no_app_exception(app):
    """
    Streamlit's AppTest.exception is an ElementList.
    An empty ElementList means no exception was captured.
    """

    assert len(app.exception) == 0, (
        f"Streamlit app raised an exception:\n{app.exception}"
    )


# ---------------------------------------------------------
# BASIC LOAD
# ---------------------------------------------------------

def test_app_loads():

    app = AppTest.from_file(str(APP_PATH))

    app.run(timeout=60)

    assert_no_app_exception(app)


# ---------------------------------------------------------
# TCS
# ---------------------------------------------------------

def test_tcs_analysis():

    app = run_app("TCS", "1y")

    assert_no_app_exception(app)


# ---------------------------------------------------------
# APPLE
# ---------------------------------------------------------

def test_apple_analysis():

    app = run_app("Apple", "1y")

    assert_no_app_exception(app)


# ---------------------------------------------------------
# INVALID SEARCH
# ---------------------------------------------------------

def test_invalid_search():

    app = run_app("wtxyz123", "1y")

    assert_no_app_exception(app)


# ---------------------------------------------------------
# EMPTY SEARCH
# ---------------------------------------------------------

def test_empty_search():

    app = run_app("", "1y")

    assert_no_app_exception(app)


# ---------------------------------------------------------
# PERIOD TESTS
# ---------------------------------------------------------

def test_6_month_period():

    app = run_app("TCS", "6mo")

    assert_no_app_exception(app)


def test_1_year_period():

    app = run_app("TCS", "1y")

    assert_no_app_exception(app)


def test_2_year_period():

    app = run_app("TCS", "2y")

    assert_no_app_exception(app)


def test_5_year_period():

    app = run_app("TCS", "5y")

    assert_no_app_exception(app)


# ---------------------------------------------------------
# INTERNATIONAL STOCK
# ---------------------------------------------------------

def test_international_stock():

    app = run_app("Apple", "1y")

    assert_no_app_exception(app)


# ---------------------------------------------------------
# NSE STOCK
# ---------------------------------------------------------

def test_nse_stock():

    app = run_app("TCS", "1y")

    assert_no_app_exception(app)


# ---------------------------------------------------------
# EXPECTED SECTIONS
# ---------------------------------------------------------

def test_expected_sections():

    app = run_app("TCS", "1y")

    assert_no_app_exception(app)

    # Collect visible text from the app
    text = ""

    for element in app.markdown:
        text += " " + str(element.value)

    for element in app.title:
        text += " " + str(element.value)

    for element in app.subheader:
        text += " " + str(element.value)

    for element in app.caption:
        text += " " + str(element.value)

    text = text.lower()

    # The dashboard should contain at least one meaningful
    # analysis-related section.
    expected_keywords = [
        "analysis",
        "return",
        "volume",
        "market",
    ]

    assert any(keyword in text for keyword in expected_keywords), (
        "Expected analysis content was not found in the app."
    )


# ---------------------------------------------------------
# NO EXCEPTION - TCS
# ---------------------------------------------------------

def test_no_exception_for_tcs():

    app = run_app("TCS", "1y")

    assert_no_app_exception(app)


# ---------------------------------------------------------
# NO EXCEPTION - APPLE
# ---------------------------------------------------------

def test_no_exception_for_apple():

    app = run_app("Apple", "6mo")

    assert_no_app_exception(app)