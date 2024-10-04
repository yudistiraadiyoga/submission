import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from babel.numbers import format_currency
sns.set(style='dark')

def calculate_holiday_stats(hour_df):
    holiday_stats = hour_df.groupby('holiday').agg(
        avg_total_rentals=('cnt', 'mean'),
        median_total_rentals=('cnt', 'median'),
        total_rentals=('cnt', 'sum'),
        avg_casual=('casual', 'mean'),
        median_casual=('casual', 'median'),
        total_casual=('casual', 'sum'),
        avg_registered=('registered', 'mean'),
        median_registered=('registered', 'median'),
        total_registered=('registered', 'sum')
    ).reset_index()
    return holiday_stats

def calculate_season_stats(hour_df):
    season_stats = hour_df.groupby('season').agg(
        avg_total_rentals=('cnt', 'mean'),
        median_total_rentals=('cnt', 'median'),
        total_rentals=('cnt', 'sum'),
        avg_casual=('casual', 'mean'),
        median_casual=('casual', 'median'),
        total_casual=('casual', 'sum'),
        avg_registered=('registered', 'mean'),
        median_registered=('registered', 'median'),
        total_registered=('registered', 'sum')
    ).reset_index()
    return season_stats

def calculate_monthly_stats(hour_df):
    monthly_stats = hour_df.groupby('mnth').agg(
        total_rentals=('cnt', 'sum'),
        avg_rentals=('cnt', 'mean'),
        order_count=('instant', 'nunique')
    ).reset_index()
    return monthly_stats

def plot_holiday_vs_non_holiday(holiday_stats):
    max_bar = np.argmax(holiday_stats['avg_total_rentals'])
    colors = ['#FF0000' if i == max_bar else '#82CAFF' for i in range(len(holiday_stats))]
    
    plt.figure(figsize=(6, 6))
    sns.barplot(x='holiday', y='avg_total_rentals', data=holiday_stats, palette=colors)
    plt.title('Average Bike Rentals: Holidays vs Non-Holidays', fontsize=16)
    plt.xlabel('Holiday', fontsize=14)
    plt.ylabel('Average Total Rentals (cnt)', fontsize=14)
    plt.xticks(ticks=[0, 1], labels=['Non-Holiday', 'Holiday'])
    
    st.pyplot(plt)

def plot_total_rentals_holiday_vs_non_holiday(holiday_stats):
    max_bar = np.argmax(holiday_stats['total_rentals'])
    colors = ['#FF0000' if i == max_bar else '#82CAFF' for i in range(len(holiday_stats))]
    
    plt.figure(figsize=(6, 6))
    sns.barplot(x='holiday', y='total_rentals', data=holiday_stats, palette=colors)
    plt.title('Total Bike Rentals: Holidays vs Non-Holidays', fontsize=16)
    plt.xlabel('Holiday', fontsize=14)
    plt.ylabel('Total Rentals (cnt)', fontsize=14)
    plt.xticks(ticks=[0, 1], labels=['Non-Holiday', 'Holiday'])
    
    st.pyplot(plt)

def plot_avg_rentals_across_seasons(season_stats):
    max_bar = np.argmax(season_stats['avg_total_rentals'])
    colors = ['#FF0000' if i == max_bar else '#82CAFF' for i in range(len(season_stats))]
    
    plt.figure(figsize=(6, 6))
    sns.barplot(x='season', y='avg_total_rentals', data=season_stats, palette=colors)
    plt.title('Average Bike Rentals Across Seasons', fontsize=16)
    plt.xlabel('Season', fontsize=14)
    plt.ylabel('Average Total Rentals (cnt)', fontsize=14)
    plt.xticks(ticks=[0, 1, 2, 3], labels=['Spring', 'Summer', 'Fall', 'Winter'])
    
    st.pyplot(plt)

def plot_total_rentals_distribution_across_seasons(season_stats):
    season_mapping = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    season_labels = [season_mapping[season] for season in season_stats['season']]
    total_rentals_values = season_stats['total_rentals']
    
    plt.figure(figsize=(8, 8))
    explode = [0.1 if label == 'Fall' else 0 for label in season_labels]
    
    plt.pie(total_rentals_values, labels=season_labels, autopct='%1.1f%%', startangle=140,
            colors=['#82CAFF', '#FFDDC1', '#FF0000', '#FFABAB'], explode=explode)
    
    plt.title('Total Bike Rentals Distribution Across Seasons', fontsize=16)
    plt.axis('equal')
    
    st.pyplot(plt)

def plot_total_rentals_by_month(hour_df):
    month_count = hour_df.groupby('mnth')['cnt'].sum().reset_index()
    max_bar = month_count['cnt'].idxmax()
    
    colors = ['#FF0000' if i == max_bar else '#82CAFF' for i in range(len(month_count))]
    month_count.plot(
        x='mnth', y='cnt', kind='bar', figsize=(10, 5),
        color=colors, xlabel='Month', ylabel='Total Bike Rentals',
        title='Total Bike Rentals by Month (Peak Highlighted)'
    )
    
    plt.xticks(
        ticks=range(len(month_count)),
        labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(month_count)]
    )
    
    st.pyplot(plt)

hour_df = pd.read_csv('hour.csv')

datetime = ['dteday']
hour_df.sort_values(by='dteday', inplace=True)
hour_df.reset_index(inplace=True)

for column in datetime:
    hour_df[column] = pd.to_datetime(hour_df[column])

min_date = hour_df['dteday'].min()
max_date = hour_df['dteday'].max()

with st.sidebar:
    st.image('https://github.com/dicodingacademy/assets/raw/main/logo.png')
    start_date, end_date = st.date_input(
        label='Time span', min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )
main_df = hour_df[(hour_df['dteday'] >= str(start_date)) & (hour_df['dteday'] <= str(end_date))]
holiday_stats = calculate_holiday_stats(main_df)
season_stats = calculate_season_stats(main_df)
monthly_stats = calculate_monthly_stats(main_df)

st.header('Bike Sharing Dashboard')

col1, col2 = st.columns(2)

with col1:
    total_orders = monthly_stats['order_count'].sum()
    st.metric("Total Bike Rentals", value=total_orders)

with col2:
    total_revenue = monthly_stats['total_rentals'].sum()
    st.metric("Total Bike Rented Out", value=total_revenue)

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    monthly_stats["mnth"],
    monthly_stats["total_rentals"],
    marker='o', 
    linewidth=2,
    color="#90CAF9"
)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)
ax.set_xlabel('Month')
ax.set_ylabel('Total Rentals')

st.pyplot(fig)

st.subheader('Holiday vs Non Holiday')
plot_holiday_vs_non_holiday(holiday_stats)
plot_total_rentals_holiday_vs_non_holiday(holiday_stats)

st.subheader('Average Bike Rentals Across Seasons')
plot_avg_rentals_across_seasons(season_stats)

st.subheader('Total Bike Rentals Distribution Across Seasons')
plot_total_rentals_distribution_across_seasons(season_stats)

st.subheader('Total Bike Rentals by Month')
plot_total_rentals_by_month(main_df)
