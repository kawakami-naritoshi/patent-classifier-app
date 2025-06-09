import streamlit as st
import pandas as pd
import google.generativeai as genai
import io # Excelファイルデータをメモリ上で扱うために必要

# --- アプリのタイトルと説明 ---
st.set_page_config(page_title="課題分類・解決手段分類生成アプリ", layout="wide") # タイトル変更
st.title("課題分類・解決手段分類生成アプリ") # タイトル変更

# --- 機能紹介セクション ---
st.success(
    """
    **ようこそ！このアプリでできること：**

    1.  **Gemini APIキーの設定**: お持ちのGemini APIキーをサイドバーから入力します。
    2.  **Excelファイルのアップロード**: 分析したい特許情報（「要約」列を含む）が記載されたExcelファイルをアップロードします。
    3.  **分類数の指定**: 生成したい「課題」と「解決手段」の分類数を指定できます。
    4.  **要望事項の入力**: 分類を生成する際の特別な要望（例：重視する点、説明文のスタイルなど）を自由記述でAIに伝えられます。
    5.  **自動分類作成の実行**: 入力に基づいて、AIが特許文書の「課題」と「解決手段」を分析し、指定された数の分類とその説明を生成します。

    サイドバーに必要な情報を入力し、「分類を実行」ボタンを押してください。
    """
)

st.warning("""
**特許に関する重要なお知らせ**

・本プログラムは特許第7650476号の技術を使用しています。
・本プログラムにつきましては、個人的実施に留めていただけますようお願いします。
""")

# --- 入力セクション (サイドバー) ---
st.sidebar.header("設定入力")
api_key = st.sidebar.text_input("Gemini APIキー", type="password", help="Gemini APIキーを入力してください。")
uploaded_file = st.sidebar.file_uploader("分析対象のExcelファイルをアップロード", type=["xlsx", "xls"], help="「要約」列を含むExcelファイルを指定してください。")
num_classifications = st.sidebar.number_input("作成する分類の数", min_value=1, value=10, step=1, help="各課題・解決手段に対して生成する分類の数を指定します。")
user_requests = st.sidebar.text_area(
    "分類作成に関する要望事項",
    "例: 各分類の説明は100文字程度で作成してください。",
    height=150,
    help="分類生成時のプロンプトに追加される要望です。"
)

# --- メイン処理関数 ---
def process_patent_documents(api_key_input, file_data, num_classes, custom_requests):
    if not api_key_input:
        st.error("APIキーが入力されていません。サイドバーから入力してください。")
        return None, None
    if file_data is None:
        st.error("Excelファイルがアップロードされていません。サイドバーからアップロードしてください。")
        return None, None

    try:
        # API キーを設定
        genai.configure(api_key=api_key_input)
        # モデルを選択 (ユーザー指定のモデル名を使用)
        model_name = 'gemini-2.0-flash'
        model = genai.GenerativeModel(model_name)

        # Excelデータの読み込み
        excel_data = io.BytesIO(file_data.getvalue())
        df = pd.read_excel(excel_data)

        if '要約' not in df.columns:
            st.error("アップロードされたExcelファイルに「要約」列が見つかりません。")
            return None, None

        # 前処理
        df['要約'] = df['要約'].astype(str).str.replace('\n', ' ', regex=False).str.replace('\r', ' ', regex=False).str.strip()
        df_filtered = df[df['要約'].str.contains('【課題】') & df['要約'].str.contains('【解決手段】') & df['要約'].str.contains('【選択図】')].copy()

        if df_filtered.empty:
            st.warning("処理対象となる有効な特許データ（「【課題】」「【解決手段】」「【選択図】」を全て含む要約）が見つかりませんでした。")
            return "有効な課題データなし", "有効な解決手段データなし"

        df_filtered.loc[:, 'problem'] = df_filtered['要約'].str.extract('【課題】(.*?)(?=【解決手段】)', expand=False).str.strip()
        df_filtered.loc[:, 'solution'] = df_filtered['要約'].str.extract('【解決手段】(.*?)(?=(?:【選択図】|$))', expand=False).str.strip()

        df_filtered.dropna(subset=['problem', 'solution'], inplace=True)
        df_filtered = df_filtered[df_filtered['problem'].str.strip() != '']
        df_filtered = df_filtered[df_filtered['solution'].str.strip() != '']

        if df_filtered.empty:
            st.warning("課題または解決手段の抽出に失敗しました。データ形式を確認してください。")
            return "課題抽出失敗", "解決手段抽出失敗"

        p_texts = df_filtered['problem'].values
        p_combine_text = ' '.join(filter(None, p_texts))

        s_texts = df_filtered['solution'].values
        s_combine_text = ' '.join(filter(None, s_texts))

        generation_config = genai.types.GenerationConfig(
            temperature=0.1,
        )

        def generate_classifications(text_data, classification_type, num_cls, reqs):
            if not text_data or text_data.isspace():
                return f"{classification_type}データが空であるか、抽出できませんでした。"

            prompt = f"""
            以下は複数の特許の{classification_type}をまとめた文章です。
            この文章に含まれる各特許の「{classification_type}」の内容を深く分析してください。
            その上で、この文章から{num_cls}個の異なる{classification_type}分類を抽出し、それぞれの{classification_type}分類に対する説明文を生成してください。
            {classification_type}分類は、{classification_type}内容をわかりやすく分類するもので、以下のように行います。

            Steps:
            1. 与えられた文章から、{classification_type}の主要な要素やテーマを抽出してください。
            2. 抽出した要素をもとに、内容の重複を避けつつ、包括的かつ特有な観点から{num_cls}個の異なる{classification_type}分類を考案し、カテゴリ名として提案してください。
            3. 各{classification_type}分類に対して、日本語で簡潔かつ具体的に説明文を記述してください。{classification_type}の本質が理解できるように記述してください。

            User Requests:
            {reqs}

            Notes:
            - 各{classification_type}分類のタイトル（カテゴリ名）は、与えられた文章の内容を正確かつ端的に反映するようにしてください。
            - 説明文は具体的であり、{classification_type}の内容を読みやすく伝えるように工夫してください。
            - {classification_type}分類が互いに重複しないように注意し、幅広い視点から分類を行ってください。
            - 出力形式は、各分類のタイトルとその説明を明確に区別できるように以下の形式で出力してください。
            [分類タイトル]
            説明文...
            [分類タイトル]
            説明文...

            以下が対象の文章です:
            {text_data}
            """
            try:
                response = model.generate_content(
                    [prompt],
                    generation_config=generation_config
                )
                return response.text
            except Exception as e:
                st.error(f"{classification_type}分類の生成中にエラーが発生しました（モデル: {model_name}）。エラー詳細: {str(e)}")
                st.info("指定されたモデル名が利用可能か、またはAPIキーに問題がないか確認してください。")
                return f"{classification_type}分類の生成に失敗しました。エラーメッセージを確認してください。"


        problem_classification_result = "課題データが不足しているため、分類を生成できませんでした。"
        if p_combine_text and not p_combine_text.isspace():
            with st.spinner(f"{num_classes}個の課題分類を生成中 (モデル: {model_name})..."):
                problem_classification_result = generate_classifications(p_combine_text, "課題", num_classes, custom_requests)
        else:
            st.info("抽出された課題テキストが空か、または有効な内容が含まれていません。課題分類は生成されません。")


        solution_classification_result = "解決手段データが不足しているため、分類を生成できませんでした。"
        if s_combine_text and not s_combine_text.isspace():
            with st.spinner(f"{num_classes}個の解決手段分類を生成中 (モデル: {model_name})..."):
                solution_classification_result = generate_classifications(s_combine_text, "解決手段", num_classes, custom_requests)
        else:
            st.info("抽出された解決手段テキストが空か、または有効な内容が含まれていません。解決手段分類は生成されません。")

        return problem_classification_result, solution_classification_result

    except Exception as e:
        st.error(f"処理全体でエラーが発生しました: {e}")
        return f"エラー: {str(e)}", f"エラー: {str(e)}"

# --- 実行ボタンと結果表示 ---
if st.sidebar.button("分類を実行", type="primary"):
    if not api_key: # APIキーが未入力の場合に再度確認
        st.error("APIキーが入力されていません。サイドバーから入力してください。")
    elif not uploaded_file: # ファイルが未アップロードの場合に再度確認
        st.error("Excelファイルがアップロードされていません。サイドバーからアップロードしてください。")
    else:
        problem_result, solution_result = process_patent_documents(api_key, uploaded_file, num_classifications, user_requests)

        if problem_result or solution_result:
            st.subheader("課題分類作成の結果") # 文言変更
            if problem_result:
                st.markdown(problem_result)
            else:
                st.info("課題分類作成の結果はありません。")

            st.markdown("---")

            st.subheader("解決手段分類作成の結果") # 文言変更
            if solution_result:
                st.markdown(solution_result)
            else:
                st.info("解決手段作成の結果はありません。")
else:
    # 初期表示時、またはボタンが押されていない場合
    if not api_key and not uploaded_file :
         st.info("サイドバーでAPIキー、Excelファイル、分類数、要望事項を設定し、「分類を実行」ボタンを押してください。")
    elif not api_key:
        st.warning("APIキーをサイドバーから入力してください。")
    elif not uploaded_file:
        st.warning("Excelファイルをサイドバーからアップロードしてください。")
    else:
        st.info("準備ができましたら、サイドバーの「分類を実行」ボタンを押してください。")


st.sidebar.markdown("---")
st.sidebar.caption("Powered by Google Gemini")
st.sidebar.caption(f"Using model: gemini-2.0-flash")
st.sidebar.caption("Ⓒ2025")
