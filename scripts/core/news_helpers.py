# 📰 ZARGUL NEWS HELPER (YOUR CUSTOM SETTINGS)
# ⚠️ SAFE ZONE - WON'T TOUCH EXISTING CODE ⚠️

# ======================
# 🎯 YOUR KEYWORD LIST
# ======================
KEYWORDS = [
    # Economic Factors
    "inflation",
    "interest rate hike",
    "recession",
    "oil prices",
    "supply chain",
    "tariffs",
    "sanctions",
    "rate cuts",
    "stimulus",
    "unemployment",
    "economic slowdown",
    "fed tightening",
    "housing bubble",
    "default",
    "debt ceiling",
    "trade war",
    "geopolitical risk",
    "war",
    "nuclear threat",
    "oil embargo",
    "fed meeting",
    "inflation data",
    "gdp contraction",
    "interest rates",
    "yield curve",
    "supply shortage",
    "commodity prices",
    "devaluation",
    "energy crisis",
    "black swan event",

    # Crypto Specific
    "crypto regulation",
    "whale alert",
    "market crash",
    "bitcoin",
    "ethereum",
    "bnb",
    "solana",
    "ada",
    "doge",
    "usdt",
    "etf approval",
    "bank collapse"
]

# ======================
# 🕵️ YOUR PEOPLE WATCHLIST
# ======================
PEOPLE_WATCH = {
    # Influential Individuals
    "Elon Musk": 9,
    "Donald Trump": 9,
    "Senator Josh Hawley": 7,
    "Mike Novogratz": 8,
    "Cathie Wood": 8,
    "Michael Saylor": 9,
    "Jerome Powell": 10,
    "Christine Lagarde": 9,

    # News Sources
    "Bloomberg News": 10,
    "Reuters": 9,
    "Financial Times": 9,
    "CNBC": 8,
    "Wall Street Journal": 9,
    "ZeroHedge": 7,

    # Crypto Trackers
    "Whale Alert": 8
}

# ======================
# 🚀 IMPACT BOOST VALUES
# ======================
BOOST_VALUES = {
    # People Impacts
    "Jerome Powell": 0.15,  # +15% confidence
    "Elon Musk": 0.12,  # +12% confidence
    "Christine Lagarde": 0.10,  # +10% confidence

    # Economic Terms
    "interest rate hike": -0.25,  # -25% confidence
    "recession": -0.30,
    "market crash": -0.35,

    # Positive Crypto News
    "etf approval": 0.30,  # +30% confidence
    "rate cuts": 0.15,
    "stimulus": 0.12,

    # Negative Events
    "bank collapse": -0.40,
    "sanctions": -0.20,
    "crypto regulation": -0.15,

    # Extreme Events
    "black swan event": -0.50,  # -50% confidence
    "war": -0.45,
    "nuclear threat": -0.60
}

# ======================
# 📊 NEWS CATEGORIES
# ======================
NEWS_CATEGORIES = {
    "EXTREME_NEGATIVE": ["war", "nuclear threat", "black swan event"],
    "REGULAR_NEGATIVE": ["recession", "market crash", "bank collapse"],
    "NEUTRAL": ["fed meeting", "inflation data", "gdp contraction"],
    "POSITIVE": ["etf approval", "rate cuts", "stimulus"],
    "CRYPTO_SPECIFIC": ["bitcoin", "ethereum", "whale alert"]
}


# ======================
# 🛡️ SAFETY CHECKS
# ======================
def is_valid_news(source, content):
    """Check if news is from trusted source and contains real content"""
    trusted_sources = [
        "Bloomberg News", "Reuters", "Financial Times", "Wall Street Journal"
    ]
    return (source in trusted_sources) and (len(content) > 50)
