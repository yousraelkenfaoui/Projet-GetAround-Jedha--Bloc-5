

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns

# Config
st.set_page_config(page_title='GetAround Project', page_icon='🚗', layout="wide", initial_sidebar_state="auto", menu_items=None)

st.title('Dashboard - Getaround Project')
st.markdown("👋 Welcome, you'll find here the dashboard created to share my analysis on the Getaround Project 🚗")
st.markdown('')

# Definition of the homepage
def main_page():
    DATA_URL = 'get_around_delay_analysis.xlsx'
    DATA_URL2 = 'get_around_pricing_project.csv'

    @st.cache_data
    def load_data():
        data = pd.read_csv(DATA_URL2)
        return data

    @st.cache_data
    def load_data2():
        data2 = pd.read_excel(DATA_URL)
        return data2

    dataset_pricing = load_data()
    df = load_data2()

    # Show dataset - Checkbox
    if st.checkbox('Show data'):
        st.write(df)

    st.header('GENERAL DATA VISUALIZATION :')

    # First visualisation
    state = df['state'].value_counts()
    pourcentages_state = state / state.sum() * 100

    fig1, ax1 = plt.subplots(figsize=(6, 2))
    ax1.pie(pourcentages_state, labels=pourcentages_state.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal') 
    ax1.set_title('Percentage of completed and cancelled reservations', fontsize=6)
    st.pyplot(fig1)

    # Second visualisation
    checkin_type = df['checkin_type'].value_counts()

    fig2, ax2 = plt.subplots(figsize=(6, 2))
    ax2.bar(checkin_type.index, checkin_type.values, color=['skyblue', 'orange'])
    ax2.set_xlabel('checkin_type')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of check-in types', fontsize=6)
    ax2.set_xticks(range(len(checkin_type)))
    ax2.set_xticklabels(checkin_type.index, rotation=45)
    st.pyplot(fig2)

    # Third visualisation
    df_mobile = df[df['checkin_type'] == 'mobile'].copy()
    df_connect = df[df['checkin_type'] == 'connect'].copy()

    df_mobile['delay_at_checkout_in_minutes'] = df_mobile['delay_at_checkout_in_minutes'].astype(float)
    df_connect['delay_at_checkout_in_minutes'] = df_connect['delay_at_checkout_in_minutes'].astype(float)

    # Drop NA
    df_mobile = df_mobile.dropna(subset=['delay_at_checkout_in_minutes'])
    df_connect = df_connect.dropna(subset=['delay_at_checkout_in_minutes'])

    conditions_mobile = [
        (df_mobile['delay_at_checkout_in_minutes'] > 0),
        (df_mobile['delay_at_checkout_in_minutes'] == 0),
        (df_mobile['delay_at_checkout_in_minutes'] < 0)
    ]
    choices = ['Delayed', 'On time', 'In advance']

    # Add a value for unknown checkout status
    df_mobile['checkout_status'] = np.select(conditions_mobile, choices, default='unknown')

    conditions_connect = [
        (df_connect['delay_at_checkout_in_minutes'] > 0),
        (df_connect['delay_at_checkout_in_minutes'] == 0),
        (df_connect['delay_at_checkout_in_minutes'] < 0)
    ]

    df_connect['checkout_status'] = np.select(conditions_connect, choices, default='Unknown')

    df_mobile['checkin_type'] = 'mobile'
    df_connect['checkin_type'] = 'connect'

    df_combined = pd.concat([df_mobile, df_connect])

    status_counts = df_combined.groupby(['checkin_type', 'checkout_status']).size().reset_index(name='Count')

    fig3, ax3 = plt.subplots(figsize=(6, 2))
    width = 0.35  
    x = np.arange(len(choices))

    # Calculate values for every check-in type 
    counts_mobile = [status_counts[(status_counts['checkin_type'] == 'mobile') & (status_counts['checkout_status'] == status)]['Count'].sum() for status in choices]
    counts_connect = [status_counts[(status_counts['checkin_type'] == 'connect') & (status_counts['checkout_status'] == status)]['Count'].sum() for status in choices]

    ax3.bar(x - width/2, counts_mobile, width, label='Mobile', color='skyblue')
    ax3.bar(x + width/2, counts_connect, width, label='Connect', color='orange')

    ax3.set_xlabel('Checkout status')
    ax3.set_ylabel('Count')
    ax3.set_title('Number of delayed, on time and in advance checkouts per Check-in type', fontsize=6)
    ax3.set_xticks(x)
    ax3.set_xticklabels(choices, rotation=45)
    ax3.legend(title='Check-in type')

    st.pyplot(fig3)

# Defintion of the second page
def page2():
    DATA_URL = 'get_around_delay_analysis.xlsx'

    st.title("DATA ANALYSIS 🚗💲")

    st.markdown("---")

    @st.cache_data
    def load_data2():
        data2 = pd.read_excel(DATA_URL)
        return data2

    df = load_data2()

    

    # Cleaning the dataset
    df_new = df.dropna(subset=['delay_at_checkout_in_minutes'])

    # Filtering positive delays
    positive_delays = df_new[df_new['delay_at_checkout_in_minutes'] > 0]['delay_at_checkout_in_minutes']
    max_positive_delay = positive_delays.max()

    # Calculating outliers
    Q1 = positive_delays.quantile(0.25)
    Q3 = positive_delays.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = positive_delays[(positive_delays < lower_bound) | (positive_delays > upper_bound)]

    # Cleaning outliers
    df_cleaned = df_new[
        (df_new['delay_at_checkout_in_minutes'] > 0) &  
        (df_new['delay_at_checkout_in_minutes'] >= lower_bound) &  
        (df_new['delay_at_checkout_in_minutes'] <= upper_bound)
    ]

    # Streamlit Dashboard
    st.header("Question 1 : Which share of our owner’s revenue would potentially be affected by the feature?")

    st.subheader("Distribution of Checkout Delays by Check-in Type")
    bins = [0, 15, 30, 60, 90, 120, float('inf')]
    labels = ['0-15 min', '15-30 min', '30 min-1h', '1h-1h30', '1h30-2h', '>2h']

    df_cleaned_mobile = df_cleaned[df_cleaned['checkin_type'] == 'mobile'].copy()
    df_cleaned_connect = df_cleaned[df_cleaned['checkin_type'] == 'connect'].copy()

    df_cleaned_mobile['delay_category'] = pd.cut(df_cleaned_mobile['delay_at_checkout_in_minutes'], bins=bins, labels=labels, right=False)
    df_cleaned_connect['delay_category'] = pd.cut(df_cleaned_connect['delay_at_checkout_in_minutes'], bins=bins, labels=labels, right=False)

    delay_category_counts_mobile = df_cleaned_mobile['delay_category'].value_counts().sort_index()
    delay_category_counts_connect = df_cleaned_connect['delay_category'].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(6, 2), sharey=True)

    axes[0].bar(delay_category_counts_mobile.index, delay_category_counts_mobile.values, color='skyblue')
    axes[0].set_title('Distribution of Checkout Delays for Mobile', fontsize=6)
    axes[0].set_xlabel('Delay category')
    axes[0].set_ylabel('Count of reservations')
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45)
    for index, value in enumerate(delay_category_counts_mobile):
        axes[0].text(index, value + 0.5, str(value), ha='center')

    axes[1].bar(delay_category_counts_connect.index, delay_category_counts_connect.values, color='orange')
    axes[1].set_title('Distribution of Checkout Delays for Connect', fontsize=6)
    axes[1].set_xlabel('Delay category')
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45)
    for index, value in enumerate(delay_category_counts_connect):
        axes[1].text(index, value + 0.5, str(value), ha='center')

    st.pyplot(fig)
    st.markdown("For mobile, as for connect, the total number of delays under less than 2 hours is superior that the delays of more than 2 hours. But we can see that there are less delays for Connect.")
    st.markdown('')

    st.subheader("Cancellation Rate for Each Check-in Type")
    cancelled_checkins = df[df['state'] == 'canceled']['checkin_type'].value_counts()
    total_checkins_df = df['checkin_type'].value_counts()
    cancellation_rate = (cancelled_checkins / total_checkins_df) * 100
    cancellation_rate = cancellation_rate.fillna(0)

    st.write("Cancellation rate for each check-in type:")
    st.bar_chart(cancellation_rate)

    st.markdown("From the 2 graphs, we can say that Connect owners will be affected the most by the feature because they already have less delays but have more cancellation compared to Mobile owners.")
    st.markdown('')

    st.header("Question 2 : How many rentals would be affected by the feature depending on the threshold and scope we choose?")
    
    st.header("Distribution of Delays at Checkout")
    plt.figure(figsize=(6, 2))
    plt.hist(df_cleaned['delay_at_checkout_in_minutes'], bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Delays at Checkout (in minutes)', fontsize=6)
    plt.xlabel('Delay at Checkout (minutes)')
    plt.ylabel('Number of Rentals')
    plt.grid(True)
    st.pyplot(plt)

    st.markdown("The histogram above shows the distribution of delays at checkout. Most rentals have a delay of around 0 minutes, with some outliers that are significantly late.")
    st.markdown('')

    st.markdown("Let's calculate the number of rentals that would be affected based on different delay thresholds.")
    st.markdown('')

    thresholds = [0, 15, 30, 60, 120]
    affected_rentals = {}
    for threshold in thresholds:
        affected_count = df_cleaned[df_cleaned['delay_at_checkout_in_minutes'] > threshold].shape[0]
        affected_rentals[threshold] = affected_count

    affected_rentals_df = pd.DataFrame(list(affected_rentals.items()), columns=['Threshold (minutes)', 'Number of Affected Rentals'])
    st.write(affected_rentals_df)

    st.markdown("Now we have a data showing the number of rentals affected by different delay thresholds that will help us assess the impact of the feature based on the threshold chosen.")
    st.markdown('')

    st.header("Question 3 : How often are drivers late for the next check-in? How does it impact the next driver?")
    df_delay_analysis = df_cleaned[(df_cleaned['time_delta_with_previous_rental_in_minutes'].notna()) & 
                        (df_cleaned['delay_at_checkout_in_minutes'].notna())]

    impacted_rentals = df_delay_analysis[df_delay_analysis['delay_at_checkout_in_minutes'] >= 
                                        df_delay_analysis['time_delta_with_previous_rental_in_minutes']]

    num_impacted_rentals = impacted_rentals.shape[0]
    percentage_impacted_rentals = (num_impacted_rentals / df_delay_analysis.shape[0]) * 100

    st.write(f"Number of impacted rentals: {num_impacted_rentals}")
    st.write(f"Percentage of impacted rentals: {percentage_impacted_rentals:.2f}%")

    plt.figure(figsize=(6, 2))
    plt.bar(['Impacted Rentals', 'Non-Impacted Rentals'], 
            [num_impacted_rentals, df_delay_analysis.shape[0] - num_impacted_rentals], 
            color=['tomato', 'skyblue'])
    plt.title('Impact of Late Check-outs on Next Rentals', fontsize=6)
    plt.ylabel('Number of Rentals')
    plt.grid(True)
    st.pyplot(plt)

    st.markdown("Out of the rentals where the timing between rentals is relevant, approximately 213 rentals (or 29.05%) experienced a delay at checkout that likely impacted the next driver.")
    st.markdown('')

    st.header("Question 4 : How many problematic cases will it solve depending on the chosen threshold and scope?")
    solved_cases = {}
    for threshold in thresholds:
        solved_count = impacted_rentals[impacted_rentals['delay_at_checkout_in_minutes'] <= threshold].shape[0]
        solved_cases[threshold] = solved_count

    solved_cases_df = pd.DataFrame(list(solved_cases.items()), columns=['Threshold (minutes)', 'Problematic Cases Solved'])
    st.write(solved_cases_df)

    plt.figure(figsize=(6, 2))
    plt.plot(solved_cases_df['Threshold (minutes)'], solved_cases_df['Problematic Cases Solved'], marker='o', color='green')
    plt.title('Problematic Cases Solved by Threshold', fontsize=6)
    plt.xlabel('Threshold (minutes)')
    plt.ylabel('Number of Problematic Cases Solved')
    plt.grid(True)
    st.pyplot(plt)

    st.header("CONCLUSION :")
    st.markdown("This information will help you understand the potential effectiveness of the chosen threshold in resolving the issue of late checkouts.")
    st.markdown("But the choice of the threshold really depends on the company's goals and expectations, do they want to keep more rentals possible or play on improving the impacts caused by vehicle returns.")
    st.markdown("Also, if the delay increases > the turnover will decrease.")
    st.markdown('')


# Page choice
page = st.sidebar.selectbox("Choose a page", ["Homepage", "Analysis"])

if page == "Homepage":
    main_page()
else:
    page2()
