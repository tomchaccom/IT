"""RSS 피드·매칭 임계값 등 런타임 상수."""

MATCH_THRESHOLD = 0.12

DEFAULT_FEEDS: tuple[str, ...] = (
    "https://news.ycombinator.com/rss",
    "https://techcrunch.com/feed/",
    "https://www.theverge.com/rss/index.xml",
)

EXTENDED_FEEDS: tuple[str, ...] = (
    "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "https://www.wired.com/feed/tag/ai/latest/rss",
)

HTTP_TIMEOUT = 20.0
USER_AGENT = "ITNewsDigestBot/1.0 (+https://github.com/langchain-ai/langgraph)"

# NewsAPI.org — https://newsapi.org/docs
# 환경 변수: NEWS_API_KEY (또는 호환용 NEWSAPI_API_KEY). 없으면 RSS만 수집.
NEWSAPI_TOP_HEADLINES_URL = "https://newsapi.org/v2/top-headlines"
NEWSAPI_PAGE_SIZE = 15
