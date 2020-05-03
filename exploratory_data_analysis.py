import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
from textblob import TextBlob
from nlp_preprocessing import text_preprocessing
from nlp_processing import get_ngrams, TopicModeling

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)


def check_missing_val(df):
    """print missing entries chart"""
    if len(df[df.isnull().any(axis=1)] != 0):
        print("\nPreview of data with null values:")
        print(df[df.isnull().any(axis=1)].head(5))
        if len(df) > 1e5:
            missingno.matrix(df.sample(10000))
        else:
            missingno.matrix(df)
        plt.show()


def check_duplicates(df):
    """check for duplicate"""
    dup = df.duplicated()
    if len(df[dup]) > 0:
        print(f"\nNumber of duplicated entries: {len(df[dup])}")
        print(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)).head(10))
    else:
        print("\nNo duplicated entries found")


def top5(df):
    """
    :param df: pandas DataFrame
    generate top-5 unique values for non-numeric data
    """
    columns = df.select_dtypes(include=['object', 'category']).columns
    for col in columns:
        print(f"\nTop-5 unique values of {col}")
        tmp = df[col].value_counts().reset_index()
        print(tmp.rename(columns={"index": col, col: 'Count'})[:min(5, len(tmp))])


def sentiment_analysis(df, col):
    print("\nPolarity sentiment analysis:")
    df[f'{col}_polarity'] = df[col].map(lambda text: TextBlob(text).sentiment.polarity)

    print('\ntop-5 positive reviews:')
    mx_polarity = df.nlargest(5, f'{col}_polarity')
    for ind in mx_polarity.index:
        print(f"{mx_polarity.loc[ind, f'{col}_polarity']:.2f} {mx_polarity.loc[ind, col]}")

    print('\ntop-5 negative reviews:')
    mn_polarity = df.nsmallest(5, f'{col}_polarity')
    for ind in mn_polarity.index:
        print(f"{mn_polarity.loc[ind, f'{col}_polarity']:.2f} {mn_polarity.loc[ind, col]}")

    print('\n5 random neural reviews:')
    nu_polarity = df[df[f'{col}_polarity'] == 0].sample(5)
    for ind in nu_polarity.index:
        print(f"{nu_polarity.loc[ind, f'{col}_polarity']:.2f} {nu_polarity.loc[ind, col]}")
    _, ax = plt.subplots()
    df[f'{col}_polarity'].plot.hist(bins=50, ax=ax, title='Sentiment Polarity Distribution')


def ngrams_analysis(df, col, topk=20):
    for i, n in zip(range(1, 4), ['Uni', 'Bi', 'Tri']):
        freq_words = get_ngrams(df[col], ngram_range=(i, i))
        df1 = pd.DataFrame(freq_words[:topk], columns=[col, 'count'])
        _, ax = plt.subplots()
        df1.groupby(col).sum()['count'].sort_values(ascending=False).plot.bar(rot=90, title=f'{n}gram top {topk} words')


def topic_analysis(df, col, n_topics):
    print('\nTopic Analysis:')
    topic_model = TopicModeling(n_topics)
    topic_model(df[col])

    # get top topic pairs
    lsa_categories, lsa_counts = topic_model.get_count_pairs()
    # get top-n words for each topic
    top_n_words = topic_model.get_top_n_words(5)
    for i in range(len(top_n_words)):
        print(f"Topic {i}: {top_n_words[i]}")

    # plot topic count bar
    df1 = pd.DataFrame(np.stack((lsa_categories, lsa_counts), axis=1), columns=['category', 'count'])
    _, ax = plt.subplots()
    df1.groupby('category').sum()['count'].plot.bar(title=f'LSA Topic Counts', rot=0)
    ax.set_ylabel('Number of review text')
    ax.set_xlabel('Topic')

    # T-SNE
    topic_model.plot_tsne()


def time_series_eda(df):
    """
    :param df: pandas.DataFrame
    generate times series plot of numeric data by daily, monthly and yearly frequency
    """
    datetime_cols = df.select_dtypes(include='datetime64').columns
    for col in datetime_cols:
        for p, period in zip(['D', 'M', 'Y'], ['daily', 'monthly', 'yearly']):
            print(f'\nPlotting {period} data')
            for col_num in df.select_dtypes(include=np.number).columns:
                tmp = df.copy()
                ax = tmp.set_index(col).resample(p).sum()[[col_num]].plot()
                ax.set_ylim(bottom=0)
                ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))


def numerical_eda(df, hue=None):
    """
    :param df: pandas.DataFrame
    :param hue: column in df to set as hue
    generate numerical data plots
    """
    num_columns = df.select_dtypes(include=np.number).columns
    cat_columns = df.select_dtypes(include='category').columns
    print('\nDistribution of numeric data:')
    print(df.describe().T)

    # plot univariate numerical data boxplot
    f, axes = plt.subplots(1, len(num_columns))
    for ax, col in zip(axes.flatten(), num_columns):
        sns.boxplot(y=col, data=df, ax=ax)
    f.tight_layout()

    # plot violin plot for each numerical data by category values
    if len(cat_columns) > 0:
        for col_num in num_columns:
            for col in cat_columns:
                fig = sns.catplot(x=col, y=col_num, kind='violin', data=df, height=5, aspect=2)
                fig.set_xticklabels(rotation=90)

    # plot pairwise joint distribution
    cols = list(num_columns) + [hue] if hue is not None else num_columns
    sns.pairplot(df[cols], hue=hue)


def categorical_eda(df, hue=None):
    """
    :param df: pandas.DataFrame
    :param hue: column in df to set as hue
    generate categorical data plots
    """
    columns = df.select_dtypes(include=['category']).columns
    print("\nUnique count of categorical data")
    print(df.select_dtypes(include=['object', 'category']).nunique())
    top5(df)
    # plot count distribution of categorical data
    for col in columns:
        if hue is None:
            fig = sns.catplot(x=col, kind="count", data=df, hue=hue)
            fig.set_xticklabels(rotation=90)
            for p in fig.ax.patches:
                h, w = p.get_height(), p.get_width()
                fig.ax.text(p.get_x() + w / 2., h,
                            f'{h / df[col].count():.2%}',
                            fontsize=8, ha='center', va='bottom')
        else:
            ax = pd.crosstab(df[col], df[hue]).plot.bar(stacked=True)
            ax.xaxis.set_tick_params(rotation=90)


def text_eda(df, col, polarity=True, topic=True, compare=True, **kwargs):
    """
    generate text data visualization
    :param df: pandas.DataFrame
    :param col: text column in df to explore (already preprocessed)
    :param polarity: sentiment polarity
    :param topic: topic analysis
    :param compare: scattertext comparison
    """
    assert isinstance(col, str) and col in df.columns, f'{col} should be a valid column in df'
    # sentiment analysis
    if polarity:
        sentiment_analysis(df, col)
    # text length analysis
    df[f'{col}_len'] = df[col].astype(str).apply(len)
    df[f'{col}_word_count'] = df[col].astype(str).apply(lambda x: len(x.split()))
    # ngrams analysis
    ngrams_analysis(df, col)
    # topic analysis
    if topic:
        n_topic = kwargs['n_topic'] if 'n_topic' in kwargs else 10
        topic_analysis(df, col, n_topic)
    if compare:
        pass
        # TODO: add scattertext visualization


def eda(df, hue=None, text=None):
    """
    generate exploratory data analysis
    :param df: pandas.DataFrame
    :param hue: column in df to set as hue
    :param text: column in df to do text eda
    """
    # check that input is pandas dataframe
    assert isinstance(df, pd.core.frame.DataFrame), "Only pandas DataFrame is allowed as input"

    # replace field that's entirely space (or empty) with NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)

    print("Preview of data:")
    print(df.head(5))

    print("\nData information:")
    print(df.info())

    # generate preview of entries with null value
    check_missing_val(df)

    # generate count statistics of duplicate entries
    check_duplicates(df)

    # EDA of text data
    if text is not None:
        text_eda(df, text)

    # EDA of categorical data
    categorical_eda(df, hue)

    # EDA of numerical data
    numerical_eda(df, hue)

    # Plot time series of numeric data
    time_series_eda(df)

    plt.show()


def test():
    # train.csv from https://www.kaggle.com/c/rossmann-store-sales/data
    df = pd.read_csv('train.csv')
    # set identifier "Store" as string
    df['Store'] = df['Store'].astype('str')
    # set categorical data
    df['DayOfWeek'] = df['DayOfWeek'].astype('category')
    df['Open'] = df['Open'].astype('category')
    df['Promo'] = df['Promo'].astype('category')
    df['StateHoliday'] = df['StateHoliday'].astype(str).str.strip().astype('category')
    df['SchoolHoliday'] = df['SchoolHoliday'].astype('category')
    # set datetime data
    df['Date'] = pd.to_datetime(df['Date'])

    # test full eda
    eda(df)

    # check hue
    categorical_eda(df, hue='DayOfWeek')
    numerical_eda(df, hue='DayOfWeek')

    # check dup
    tmp = df.append(df.loc[np.random.randint(0, len(df), 3)]).reset_index(drop=True)
    check_duplicates(tmp)

    # check missing val
    df.loc[np.random.randint(0, len(df), 50000), 'DayOfWeek'] = np.nan
    check_missing_val(df.sample(10000))

    plt.show()


def test2():
    # data set: https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews
    df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv', index_col=0)
    df['Rating'] = df['Rating'].astype('category')
    df['Recommended IND'] = df['Recommended IND'].astype('category')

    col = 'Review Text'
    # preprocess: do once
    if 0:
        df.drop('Title', axis=1, inplace=True)
        df = df[df[col].notna()]
        # preprocess text
        df[col] = df[col].astype(str).apply(lambda text: " ".join(text_preprocessing(text)))
        df.to_csv('Womens Clothing E-Commerce Reviews.csv')
    eda(df, text=col)


if __name__ == '__main__':
    # test()
    test2()
