#!/usr/bin/env python3
import os
import sys
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import urllib.parse
from datetime import datetime

# ==================== 情報収集セクション ====================

def fetch_github_trending(language="python"):
    url = f"https://github.com/trending/{language}?since=daily"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    print(f"[*] GitHub Trending ({language}) の情報収集中...")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        repos = soup.select('article.Box-row')
        
        results = []
        for repo in repos[:10]:
            title_el = repo.select_one('h2 a')
            repo_name = title_el.text.strip().replace('\n', '').replace(' ', '') if title_el else "Unknown"
            repo_url = "https://github.com" + title_el['href'] if title_el else ""
            
            desc_el = repo.select_one('p')
            description = desc_el.text.strip() if desc_el else "説明なし"
            
            stars_today_el = repo.select_one('span.d-inline-block.float-sm-right')
            stars_today = stars_today_el.text.strip() if stars_today_el else "N/A"
            
            stars_total_el = repo.select('a.Link--muted')
            stars_total = "N/A"
            if len(stars_total_el) > 0:
                stars_total = stars_total_el[0].text.strip()
                
            results.append({
                'name': repo_name,
                'url': repo_url,
                'description': description,
                'stars_today': stars_today,
                'stars_total': stars_total
            })
        return results
    except Exception as e:
        print(f"[!] GitHub Trendingの取得に失敗しました: {e}", file=sys.stderr)
        return []

def fetch_google_news(query, limit=10):
    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    
    print(f"[*] Google News RSS (クエリ: {query}) の情報収集中...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        items = root.findall('.//item')
        
        results = []
        for item in items[:limit]:
            title = item.find('title').text if item.find('title') is not None else "No Title"
            link = item.find('link').text if item.find('link') is not None else ""
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
            source = item.find('source').text if item.find('source') is not None else "Unknown"
            
            results.append({
                'title': title,
                'link': link,
                'pub_date': pub_date,
                'source': source
            })
        return results
    except Exception as e:
        print(f"[!] Google News RSSの取得に失敗しました: {e}", file=sys.stderr)
        return []

# ==================== LLM 処理セクション ====================

def call_llm(prompt):
    """環境変数に応じてGeminiまたはOpenAI APIを呼び出す"""
    # 1. Gemini APIの確認
    if os.environ.get("GEMINI_API_KEY"):
        gemini_key = os.environ.get("GEMINI_API_KEY")
        print(f"[*] Gemini APIキーを検出しました (先頭4文字: {gemini_key[:4]}...)。")
        try:
            # フリーズを避けるため、同期的な google-generativeai を使用
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            
            # 最新の推奨モデルを指定
            print("[*] Gemini 3.5 Flashモデルでの生成を開始します...")
            model = genai.GenerativeModel('gemini-3.5-flash')
            
            # 60秒のタイムアウトを設定し、フリーズを防止
            response = model.generate_content(
                prompt,
                request_options={"timeout": 60.0}
            )
            return response.text
        except Exception as e:
            print(f"[!] Gemini APIの呼び出し中にエラーが発生しました: {e}", file=sys.stderr)
            return None

    # 2. OpenAI APIの確認
    elif os.environ.get("OPENAI_API_KEY"):
        openai_key = os.environ.get("OPENAI_API_KEY")
        print(f"[*] OpenAI APIキーを検出しました (先頭4文字: {openai_key[:4]}...)。呼び出しを開始します。")
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[!] OpenAI APIの呼び出し中にエラーが発生しました: {e}", file=sys.stderr)
            return None
        
    else:
        print("[!] エラー: GEMINI_API_KEY または OPENAI_API_KEY が環境変数に設定されていません。", file=sys.stderr)
        return None

def generate_contents(raw_data_text):
    """収集したトレンドデータからレポートとSNS用テキストを生成するプロンプトを作成"""
    
    # 1. レポート用のプロンプト
    report_prompt = f"""あなたは最先端の「AIビジネス・トレンドアナリスト」です。以下の収集された最新トレンドデータ（GitHubの急上昇リポジトリ、GoogleニュースのAI/SaaS関連情報）に基づき、深く洞察された日本語の週刊インテリジェンス・レポートを執筆してください。

単なる情報の要約にとどまらず、以下の3点に重点を置いてください：
- **トレンドの本質（インサイト）**: テクノロジーやビジネスがどの方向に向かっているか。
- **ソロプレナー（個人開発者）への影響**: この動きが、少ないリソースで起業する人々にどう寄与するか。
- **具体的なビジネスチャンス**: どのような製品やサービスを作ればブルーオーシャンで戦えるか。

データ：
```text
{raw_data_text}
```

レポートの形式：
Markdown形式で出力してください。見やすい見出し、表、箇条書きなどを適宜使用してください。"""

    # 2. SNSポスト用のプロンプト
    sns_prompt_template = """あなたはSNSのマーケティングエキスパートです。以下の生成されたトレンドレポートの要点を凝縮し、X（旧Twitter）で拡散されやすいスレッド投稿（全5〜6ポスト）の下書きを作成してください。

ターゲット：AI開発者、インディハッカー、個人開発者、副業・起業に関心がある人。

ルール：
- フックが強く、リツイートやいいねを誘う文章であること。
- 各ポストの終わりに (1/5) (2/5) のようにスレッド番号を入れること。
- 最終ポストは、レポート全文へのリンク（[レポートを読む] プレースホルダー）を配置すること。

レポート内容：
{report_placeholder}"""

    # レポート生成
    print("[*] レポートの生成をリクエスト中...")
    report_text = call_llm(report_prompt)
    
    if not report_text:
        print("[!] レポート生成結果が空（None）です。")
        return None, None
        
    # レポートを元にSNSポスト生成
    print("[*] SNS告知文の生成をリクエスト中...")
    sns_prompt = sns_prompt_template.replace("{report_placeholder}", report_text)
    sns_text = call_llm(sns_prompt)
    
    return report_text, sns_text

# ==================== メイン実行セクション ====================

def main():
    # 1. データの収集
    github_repos = fetch_github_trending("python")
    google_news_ai = fetch_google_news('"AI agent" OR "AI agents" OR "AI automation"', limit=10)
    google_news_saas = fetch_google_news('"micro-SaaS" OR "indie hacker" OR "solopreneur"', limit=5)
    
    raw_content = []
    raw_content.append(f"=== Trend Collection Report (Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
    
    raw_content.append("## GitHub Trending (Python)\n")
    for idx, repo in enumerate(github_repos, 1):
        raw_content.append(f"{idx}. {repo['name']}\n   - URL: {repo['url']}\n   - Description: {repo['description']}\n   - Stats: {repo['stars_today']} (Total: {repo['stars_total']})")
    
    raw_content.append("\n## Google News: AI Agents / AI Automation\n")
    for idx, news in enumerate(google_news_ai, 1):
        raw_content.append(f"{idx}. {news['title']}\n   - Source: {news['source']} ({news['pub_date']})\n   - Link: {news['link']}")
        
    raw_content.append("\n## Google News: Micro-SaaS / Solopreneur\n")
    for idx, news in enumerate(google_news_saas, 1):
        raw_content.append(f"{idx}. {news['title']}\n   - Source: {news['source']} ({news['pub_date']})\n   - Link: {news['link']}")
        
    raw_data_text = '\n'.join(raw_content)
    
    # 2. 実行時ディレクトリの検出（相対パスで動くように設計）
    base_dir = os.getcwd()
    os.makedirs(os.path.join(base_dir, "scratch"), exist_ok=True)
    
    # 生データの保存
    with open(os.path.join(base_dir, "scratch", "trend_raw_data.txt"), 'w', encoding='utf-8') as f:
        f.write(raw_data_text)
    print("[+] トレンド生データを保存しました。")
    
    # 3. AIコンテンツ生成の実行
    report_text, sns_text = generate_contents(raw_data_text)
    
    if report_text and sns_text:
        # レポートの保存
        with open(os.path.join(base_dir, "trend_report.md"), 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"[+] AIトレンドレポートを生成・保存しました: {os.path.join(base_dir, 'trend_report.md')}")
        
        # SNSポストの保存
        with open(os.path.join(base_dir, "scratch", "social_posts.txt"), 'w', encoding='utf-8') as f:
            f.write(sns_text)
        print(f"[+] SNS用ポスト下書きを生成・保存しました: {os.path.join(base_dir, 'scratch', 'social_posts.txt')}")
    else:
        print("[!] 警告: APIキーがないか、生成エラーが発生したため、レポートとSNSポストの保存をスキップしました。")

if __name__ == "__main__":
    main()
