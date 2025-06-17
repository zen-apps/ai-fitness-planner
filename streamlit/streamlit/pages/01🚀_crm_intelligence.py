import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Import helper functions and classes
from helpers.salesforce_helper import (
    get_salesforce_tool,
    salesforce_to_dataframe,
    create_pipeline_analysis_chart,
    check_api_connections,
    get_sample_queries,
    get_soql_reference,
)

from helpers.salesforce_llm_helper import (
    generate_contact_insights,
    generate_account_analysis,
    generate_contact_sales_script,
    generate_script_audio,
)

# Configure page
st.set_page_config(
    page_title="SF Intelligence Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize session state
if "sf_tool" not in st.session_state:
    st.session_state.sf_tool = get_salesforce_tool()


def render_sidebar():
    """Render the sidebar navigation and system status."""
    with st.sidebar:
        st.header("🧭 Navigation")
        selected_page = "👥 Contact Intelligence"
        time_back_search_selected_box = st.selectbox(
            "Time Back Search",
            ["1 day", "1 week", "1 month", "1 year", "All Time"],
            index=2,  # Default to 1 month
        )
        st.session_state.time_back_search = time_back_search_selected_box

        st.markdown("---")
        st.header("⚙️ System Status")

        # Check API connections
        connections = check_api_connections()

        if connections["openai_connected"]:
            st.success("✅ OpenAI Connected")
        else:
            st.error("❌ OpenAI API Key Missing")

        if connections["salesforce_connected"]:
            st.success("✅ Salesforce Connected")
        else:
            st.error("❌ Salesforce Credentials Missing")

    return selected_page


def render_dashboard_overview():
    """Render the dashboard overview page."""
    st.header("📊 CRM Intelligence Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    # Get key metrics
    contacts_result = st.session_state.sf_tool.run(
        {"operation": "query", "query": "SELECT COUNT(Id) total FROM Contact"}
    )

    accounts_result = st.session_state.sf_tool.run(
        {"operation": "query", "query": "SELECT COUNT(Id) total FROM Account"}
    )

    opps_result = st.session_state.sf_tool.run(
        {
            "operation": "query",
            "query": "SELECT COUNT(Id) total, SUM(Amount) total_amount FROM Opportunity WHERE IsClosed = false",
        }
    )

    with col1:
        contacts_count = (
            contacts_result.get("records", [{}])[0].get("total", 0)
            if contacts_result.get("records")
            else 0
        )
        st.metric("Total Contacts", f"{contacts_count:,}")

    with col2:
        accounts_count = (
            accounts_result.get("records", [{}])[0].get("total", 0)
            if accounts_result.get("records")
            else 0
        )
        st.metric("Total Accounts", f"{accounts_count:,}")

    with col3:
        opps_count = (
            opps_result.get("records", [{}])[0].get("total", 0)
            if opps_result.get("records")
            else 0
        )
        st.metric("Open Opportunities", f"{opps_count:,}")

    with col4:
        opps_amount = (
            opps_result.get("records", [{}])[0].get("total_amount", 0)
            if opps_result.get("records")
            else 0
        )
        if opps_amount:
            st.metric("Pipeline Value", f"${opps_amount:,.0f}")
        else:
            st.metric("Pipeline Value", "$0")

    # Recent activity
    st.subheader("🔥 Recent Activity")
    recent_activity = st.session_state.sf_tool.run(
        {
            "operation": "query",
            "query": """SELECT Id, Subject, ActivityDate, WhoId, Who.Name, WhatId, What.Name 
                    FROM Task 
                    WHERE ActivityDate >= LAST_N_DAYS:7 
                    ORDER BY ActivityDate DESC 
                    LIMIT 10""",
        }
    )

    if recent_activity.get("records"):
        activity_df = salesforce_to_dataframe(recent_activity)
        st.dataframe(activity_df, use_container_width=True)
    else:
        st.info("No recent activities found")


def render_contact_intelligence():
    """Render the contact intelligence page."""
    st.header("👥 Contact Intelligence & Research")

    # Get contacts
    contacts_query = st.session_state.sf_tool.run(
        {
            "operation": "query",
            "query": """SELECT Id, Name, Email, Phone, Account.Name, Title, Department, LeadSource
                    FROM Contact 
                    WHERE Email != null 
                    ORDER BY LastModifiedDate DESC 
                    LIMIT 50""",
        }
    )

    if contacts_query.get("records"):
        contacts_df = salesforce_to_dataframe(contacts_query)

        st.subheader("🎯 Select Contact for AI Analysis")

        # Contact selection
        selected_contact_idx = st.selectbox(
            "Choose a contact to analyze:",
            range(len(contacts_df)),
            format_func=lambda x: f"{contacts_df.iloc[x]['Name']} - {contacts_df.iloc[x].get('Account_Name', 'No Company')}",
        )

        selected_contact = contacts_df.iloc[selected_contact_idx]

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📋 Contact Details")
            st.write(f"**Name:** {selected_contact['Name']}")
            st.write(f"**Email:** {selected_contact.get('Email', 'N/A')}")
            st.write(f"**Company:** {selected_contact.get('Account_Name', 'N/A')}")
            st.write(f"**Title:** {selected_contact.get('Title', 'N/A')}")
            st.write(f"**Department:** {selected_contact.get('Department', 'N/A')}")
            st.write(f"**Phone:** {selected_contact.get('Phone', 'N/A')}")

            if st.button("🔍 Generate AI Insights", type="primary"):
                with st.spinner("Researching contact and company..."):
                    insights = generate_contact_insights(
                        contact_name=selected_contact["Name"],
                        company_name=selected_contact.get(
                            "Account_Name", "Unknown Company"
                        ),
                        email=selected_contact.get("Email"),
                        time_back_search=st.session_state.get(
                            "time_back_search", "1 month"
                        ),
                    )
                    st.session_state.contact_insights = insights

        with col2:
            if "contact_insights" in st.session_state:
                insights = st.session_state.contact_insights

                st.subheader("🧠 AI-Generated Insights")

                # Contact score
                score_color = (
                    "green"
                    if insights.contact_score >= 7
                    else "orange" if insights.contact_score >= 5 else "red"
                )
                st.markdown(
                    f"**Contact Quality Score:** :{score_color}[{insights.contact_score}/10]"
                )

                # Company overview
                st.markdown("**🏢 Company Overview**")
                st.write(insights.company_overview)

                # Recent news
                if insights.recent_news:
                    st.markdown("**📰 Recent News**")
                    for news in insights.recent_news:
                        st.write(f"• {news}")

                # Business opportunities
                if insights.business_opportunities:
                    st.markdown("**💼 Business Opportunities**")
                    for opp in insights.business_opportunities:
                        st.write(f"• {opp}")

                # Engagement strategy
                if insights.engagement_strategy:
                    st.markdown("**🎯 Engagement Strategy**")
                    for strategy in insights.engagement_strategy:
                        st.write(f"• {strategy}")

                # Risk factors
                if insights.risk_factors:
                    st.markdown("**⚠️ Risk Factors**")
                    for risk in insights.risk_factors:
                        st.write(f"• {risk}")

                # Next actions
                if insights.next_actions:
                    st.markdown("**🚀 Recommended Next Actions**")
                    for action in insights.next_actions:
                        st.write(f"• {action}")

                # Sales Script Generation
                st.markdown("---")
                st.subheader("📝 Sales Script Generator")

                col_script1, col_script2 = st.columns([1, 1])

                with col_script1:
                    script_style = st.selectbox(
                        "Script Style:",
                        [
                            "Professional",
                            "Consultative",
                            "Relationship-Building",
                            "Direct",
                            "Technical",
                        ],
                        help="Choose the tone and approach for your sales script",
                    )

                    script_length = st.selectbox(
                        "Script Length:",
                        [
                            "Short (5-7 min call)",
                            "Medium (10-15 min call)",
                            "Long (20-30 min call)",
                        ],
                        index=1,
                        help="Target length for your sales conversation",
                    )

                    # TTS Options
                    with st.expander("🎙️ Audio Options", expanded=False):
                        lang_options = {
                            "English (US)": ("en", "us"),
                            "English (UK)": ("en", "co.uk"),
                            "English (AU)": ("en", "com.au"),
                            "English (CA)": ("en", "ca"),
                            "English (IN)": ("en", "co.in"),
                        }

                        selected_lang = st.selectbox(
                            "Voice Language:", list(lang_options.keys())
                        )
                        voice_speed = st.selectbox(
                            "Speech Speed:", ["Normal", "Slow", "Fast"], index=0
                        )
                        slow_speech = voice_speed == "Slow"

                with col_script2:
                    if st.button(
                        "🎯 Generate Sales Script",
                        type="primary",
                        key="generate_script",
                    ):
                        with st.spinner("Creating personalized sales script..."):
                            try:
                                script = generate_contact_sales_script(
                                    contact_name=selected_contact["Name"],
                                    contact_insights=insights,
                                    script_style=script_style,
                                    script_length=script_length,
                                    contact_details=selected_contact.to_dict(),
                                )
                                st.session_state.contact_sales_script = script
                                st.success("✅ Sales script generated successfully!")
                            except Exception as e:
                                st.error(f"Error generating sales script: {str(e)}")
                                st.info(
                                    "Please try again or check your API configuration."
                                )

                # Display Generated Script
                if "contact_sales_script" in st.session_state:
                    script = st.session_state.contact_sales_script

                    st.markdown("### 📞 Your Personalized Sales Script")

                    # Quick stats about the script
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Script Style", script_style)
                    with col_stat2:
                        st.metric("Target Length", script_length.split(" ")[0])
                    with col_stat3:
                        script_score = getattr(script, "script_effectiveness_score", 8)
                        st.metric("Effectiveness Score", f"{script_score}/10")

                    # Script content in tabs
                    tab1, tab2, tab3, tab4 = st.tabs(
                        ["📝 Full Script", "🎯 Key Points", "🎙️ Audio", "📧 Follow-up"]
                    )

                    with tab1:
                        st.markdown("#### 🚀 Opening Statement")
                        st.info(script.opening_statement)

                        st.markdown("#### 💎 Value Proposition")
                        st.success(script.value_proposition)

                        st.markdown("#### ❓ Discovery Questions")
                        for i, question in enumerate(script.discovery_questions, 1):
                            st.write(f"**{i}.** {question}")

                        st.markdown("#### 🛡️ Objection Handling")
                        for j, objection in enumerate(script.objection_handling, 1):
                            with st.expander(f"Objection {j}", expanded=False):
                                st.write(objection)

                        st.markdown("#### 🎯 Closing Statement")
                        st.warning(script.closing_statement)

                        st.markdown("#### 📞 Call to Action")
                        st.error(script.call_to_action)

                    with tab2:
                        st.markdown(f"**🎭 Recommended Tone:** {script.script_tone}")

                        st.markdown("**🎯 Key Talking Points:**")
                        for i, point in enumerate(script.key_talking_points, 1):
                            st.write(f"**{i}.** 🎯 {point}")

                        # Additional coaching tips
                        st.markdown("**💡 Coaching Tips:**")
                        coaching_tips = [
                            f"Focus on the contact's {insights.contact_score}/10 quality score",
                            "Reference recent company news to show you've done your homework",
                            "Address potential risk factors proactively",
                            "Use the discovery questions to uncover pain points",
                            "Be prepared to pivot based on their responses",
                        ]
                        for tip in coaching_tips:
                            st.write(f"• {tip}")

                    with tab3:
                        # Generate audio for the script
                        st.markdown("#### 🎙️ Audio Script Generation")

                        audio_options = st.radio(
                            "Choose audio content:",
                            [
                                "Full Script",
                                "Opening Only",
                                "Key Questions Only",
                                "Closing Only",
                            ],
                            horizontal=True,
                        )

                        # Prepare script text based on selection
                        if audio_options == "Full Script":
                            script_text = f"""
                            Hello, this is your personalized sales script for {selected_contact['Name']}.
                            
                            Opening Statement: {script.opening_statement}
                            
                            Value Proposition: {script.value_proposition}
                            
                            Here are your discovery questions: {'. '.join([f'Question {i}: {q}' for i, q in enumerate(script.discovery_questions, 1)])}
                            
                            Closing Statement: {script.closing_statement}
                            
                            Call to Action: {script.call_to_action}
                            
                            Remember to maintain a {script.script_tone.lower()} tone throughout the conversation.
                            """
                        elif audio_options == "Opening Only":
                            script_text = (
                                f"Opening Statement: {script.opening_statement}"
                            )
                        elif audio_options == "Key Questions Only":
                            script_text = f"Discovery Questions: {'. '.join([f'Question {i}: {q}' for i, q in enumerate(script.discovery_questions, 1)])}"
                        else:  # Closing Only
                            script_text = f"Closing Statement: {script.closing_statement}. Call to Action: {script.call_to_action}"

                        col_audio1, col_audio2 = st.columns([1, 1])

                        with col_audio1:
                            if st.button("🎙️ Generate Audio", key="generate_audio"):
                                with st.spinner("Creating audio version..."):
                                    lang_code, tld = lang_options[selected_lang]
                                    audio_bytes = generate_script_audio(
                                        script_text,
                                        language_code=lang_code,
                                        tld=tld,
                                        slow=slow_speech,
                                    )

                                    if audio_bytes:
                                        st.session_state.script_audio = audio_bytes
                                        st.success("🎉 Audio script generated!")

                        with col_audio2:
                            if "script_audio" in st.session_state:
                                st.download_button(
                                    label="📥 Download Audio",
                                    data=st.session_state.script_audio,
                                    file_name=f"sales_script_{selected_contact['Name'].replace(' ', '_')}_{audio_options.replace(' ', '_').lower()}.mp3",
                                    mime="audio/mp3",
                                    key="download_audio",
                                )

                        if "script_audio" in st.session_state:
                            st.audio(st.session_state.script_audio, format="audio/mp3")

                            # Audio settings info
                            st.info(
                                f"🎙️ Generated with {selected_lang} voice at {voice_speed.lower()} speed"
                            )

                    with tab4:
                        st.markdown("#### 📧 Follow-up Planning")

                        # Generate follow-up suggestions
                        follow_up_timeline = [
                            "📅 **Immediate (Same Day)**: Send connection request on LinkedIn with personalized message",
                            "📅 **Day 2-3**: Send follow-up email with relevant case study or resource",
                            "📅 **Week 1**: Schedule demo or discovery call if initial interest shown",
                            "📅 **Week 2**: Share industry insights or relevant news about their company",
                            "📅 **Month 1**: Quarterly business review or check-in call",
                        ]

                        for follow_up in follow_up_timeline:
                            st.write(follow_up)

                        st.markdown("#### 📝 Follow-up Email Templates")

                        email_template = f"""
**Subject**: Great connecting with you, {selected_contact['Name']}

Hi {selected_contact['Name']},

Thank you for taking the time to speak with me today about {selected_contact.get('Account_Name', 'your company')}'s challenges and goals.

Based on our conversation, I believe there are some exciting opportunities to help with [specific pain point discussed].

Next steps:
• [Specific action item from call]
• [Resource to share]
• [Proposed meeting/demo]

I'll follow up early next week to schedule our next conversation.

Best regards,
[Your Name]
                        """

                        st.text_area(
                            "Email Template:",
                            value=email_template,
                            height=200,
                            key="email_template",
                        )

                        col_email1, col_email2 = st.columns(2)
                        with col_email1:
                            st.download_button(
                                label="📧 Download Email Template",
                                data=email_template,
                                file_name=f"follow_up_email_{selected_contact['Name'].replace(' ', '_')}.txt",
                                mime="text/plain",
                                key="download_email",
                            )

                        with col_email2:
                            if st.button("📋 Copy to Clipboard", key="copy_email"):
                                st.info("Email template ready to copy!")

                # Export Options Section
                st.markdown("---")
                st.markdown("### 📥 Export Complete Package")

                col_export1, col_export2, col_export3 = st.columns(3)

                with col_export1:
                    # Comprehensive text export - only if script exists
                    if "contact_sales_script" in st.session_state:
                        script = st.session_state.contact_sales_script
                        comprehensive_script = f"""
COMPLETE SALES PACKAGE FOR: {selected_contact['Name']}
COMPANY: {selected_contact.get('Account_Name', 'N/A')}
TITLE: {selected_contact.get('Title', 'N/A')}
EMAIL: {selected_contact.get('Email', 'N/A')}
PHONE: {selected_contact.get('Phone', 'N/A')}

GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M')}
SCRIPT STYLE: {script_style}
TARGET LENGTH: {script_length}

=== AI INSIGHTS ===
Contact Quality Score: {insights.contact_score}/10
Company Overview: {insights.company_overview}

Business Opportunities:
{chr(10).join([f"• {opp}" for opp in insights.business_opportunities])}

Risk Factors:
{chr(10).join([f"• {risk}" for risk in insights.risk_factors])}

=== SALES SCRIPT ===

OPENING STATEMENT:
{script.opening_statement}

VALUE PROPOSITION:
{script.value_proposition}

DISCOVERY QUESTIONS:
{chr(10).join([f"{i}. {q}" for i, q in enumerate(script.discovery_questions, 1)])}

OBJECTION HANDLING:
{chr(10).join([f"• {o}" for o in script.objection_handling])}

CLOSING STATEMENT:
{script.closing_statement}

CALL TO ACTION:
{script.call_to_action}

RECOMMENDED TONE: {script.script_tone}

KEY TALKING POINTS:
{chr(10).join([f"• {p}" for p in script.key_talking_points])}

=== FOLLOW-UP PLAN ===
{chr(10).join(follow_up_timeline)}
"""

                        st.download_button(
                            label="📄 Download Complete Package",
                            data=comprehensive_script,
                            file_name=f"complete_sales_package_{selected_contact['Name'].replace(' ', '_')}.txt",
                            mime="text/plain",
                            key="download_complete",
                        )
                    else:
                        st.info(
                            "Generate a sales script first to download the complete package"
                        )

                with col_export2:
                    if st.button("📊 Generate CRM Task", key="create_task"):
                        # This would integrate with Salesforce to create a task
                        task_description = f"Follow up on sales conversation with {selected_contact['Name']} at {selected_contact.get('Account_Name', 'Unknown')}. Reference generated sales script and insights."
                        st.info(
                            "CRM task creation would integrate with your Salesforce API"
                        )
                        st.code(f"Task: {task_description}")

                with col_export3:
                    if st.button("📋 Add to Sales Sequence", key="add_sequence"):
                        st.info(
                            "Integration with sales sequence automation would be implemented here"
                        )


def render_account_analysis():
    """Render the account analysis page."""
    st.header("🏢 Account Intelligence & Analysis")

    # Get accounts
    accounts_query = st.session_state.sf_tool.run(
        {
            "operation": "query",
            "query": """SELECT Id, Name, Industry, Type, AnnualRevenue, NumberOfEmployees, 
                    BillingCountry, Website, Description
                    FROM Account 
                    WHERE Name != null 
                    ORDER BY LastModifiedDate DESC 
                    LIMIT 30""",
        }
    )

    if accounts_query.get("records"):
        accounts_df = salesforce_to_dataframe(accounts_query)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("🎯 Select Account")
            selected_account_idx = st.selectbox(
                "Choose an account to analyze:",
                range(len(accounts_df)),
                format_func=lambda x: accounts_df.iloc[x]["Name"],
            )

            selected_account = accounts_df.iloc[selected_account_idx]

            st.write(f"**Name:** {selected_account['Name']}")
            st.write(f"**Industry:** {selected_account.get('Industry', 'N/A')}")
            st.write(f"**Type:** {selected_account.get('Type', 'N/A')}")
            st.write(
                f"**Revenue:** ${selected_account.get('AnnualRevenue', 0):,}"
                if selected_account.get("AnnualRevenue")
                else "**Revenue:** N/A"
            )
            st.write(
                f"**Employees:** {selected_account.get('NumberOfEmployees', 'N/A')}"
            )
            st.write(f"**Country:** {selected_account.get('BillingCountry', 'N/A')}")

            if st.button("📊 Generate Account Analysis", type="primary"):
                with st.spinner("Analyzing account and market position..."):
                    analysis = generate_account_analysis(
                        account_name=selected_account["Name"],
                        account_data=selected_account.to_dict(),
                        time_back_search=st.session_state.get(
                            "time_back_search", "1 month"
                        ),
                    )
                    st.session_state.account_analysis = analysis

        with col2:
            if "account_analysis" in st.session_state:
                analysis = st.session_state.account_analysis

                st.subheader("📈 Account Intelligence Report")

                # Growth potential score
                growth_color = (
                    "green"
                    if analysis.growth_potential >= 7
                    else "orange" if analysis.growth_potential >= 5 else "red"
                )
                st.markdown(
                    f"**Growth Potential:** :{growth_color}[{analysis.growth_potential}/10]"
                )

                # Company health
                st.markdown("**🏥 Company Health**")
                st.write(analysis.company_health)

                # Market position
                st.markdown("**📍 Market Position**")
                st.write(analysis.market_position)

                # Competitive landscape
                if analysis.competitive_landscape:
                    st.markdown("**🥊 Key Competitors**")
                    for competitor in analysis.competitive_landscape:
                        st.write(f"• {competitor}")

                # Expansion opportunities
                if analysis.expansion_opportunities:
                    st.markdown("**🚀 Expansion Opportunities**")
                    for opp in analysis.expansion_opportunities:
                        st.write(f"• {opp}")

                # Retention risks
                if analysis.retention_risks:
                    st.markdown("**⚠️ Retention Risks**")
                    for risk in analysis.retention_risks:
                        st.write(f"• {risk}")

        # Account metrics visualization
        st.subheader("📊 Account Portfolio Overview")

        if not accounts_df.empty:
            # Industry distribution
            col1, col2 = st.columns(2)

            with col1:
                if "Industry" in accounts_df.columns:
                    industry_counts = accounts_df["Industry"].value_counts()
                    fig = px.pie(
                        values=industry_counts.values,
                        names=industry_counts.index,
                        title="Accounts by Industry",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if "Type" in accounts_df.columns:
                    type_counts = accounts_df["Type"].value_counts()
                    fig = px.bar(
                        x=type_counts.index,
                        y=type_counts.values,
                        title="Accounts by Type",
                    )
                    st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No accounts found or unable to connect to Salesforce")


def render_pipeline_analytics():
    """Render the pipeline analytics page."""
    st.header("📈 Pipeline Analytics & Forecasting")

    # Get opportunities
    opps_query = st.session_state.sf_tool.run(
        {
            "operation": "query",
            "query": """SELECT Id, Name, StageName, Amount, Probability, CloseDate, 
                    Account.Name, Owner.Name, LeadSource, Type
                    FROM Opportunity 
                    WHERE Amount > 0 
                    ORDER BY CloseDate ASC 
                    LIMIT 100""",
        }
    )

    if opps_query.get("records"):
        opps_df = salesforce_to_dataframe(opps_query)

        # Convert CloseDate to datetime
        if "CloseDate" in opps_df.columns:
            opps_df["CloseDate"] = pd.to_datetime(opps_df["CloseDate"])

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        total_pipeline = opps_df["Amount"].sum()
        avg_deal_size = opps_df["Amount"].mean()
        weighted_pipeline = (opps_df["Amount"] * opps_df["Probability"] / 100).sum()

        with col1:
            st.metric("Total Pipeline", f"${total_pipeline:,.0f}")
        with col2:
            st.metric("Weighted Pipeline", f"${weighted_pipeline:,.0f}")
        with col3:
            st.metric("Average Deal Size", f"${avg_deal_size:,.0f}")
        with col4:
            avg_probability = opps_df["Probability"].mean()
            st.metric("Avg Win Probability", f"{avg_probability:.1f}%")

        # Pipeline analysis chart
        if len(opps_df) > 0:
            fig = create_pipeline_analysis_chart(opps_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        # Opportunities table with filters
        st.subheader("🎯 Opportunity Pipeline")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            stages = opps_df["StageName"].unique()
            selected_stages = st.multiselect("Filter by Stage", stages, default=stages)

        with col2:
            min_amount = st.number_input("Minimum Amount", value=0, step=1000)

        with col3:
            min_probability = st.slider("Minimum Probability", 0, 100, 0)

        # Apply filters
        filtered_df = opps_df[
            (opps_df["StageName"].isin(selected_stages))
            & (opps_df["Amount"] >= min_amount)
            & (opps_df["Probability"] >= min_probability)
        ]

        st.dataframe(filtered_df, use_container_width=True)

    else:
        st.warning("No opportunities found or unable to connect to Salesforce")


def render_custom_query_builder():
    """Render the custom query builder page."""
    st.header("🔍 Custom SOQL Query Builder")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("✍️ Write Your Query")

        # Sample queries
        sample_queries = get_sample_queries()

        selected_sample = st.selectbox(
            "📋 Choose a sample query:", ["Custom"] + list(sample_queries.keys())
        )

        if selected_sample != "Custom":
            default_query = sample_queries[selected_sample]
        else:
            default_query = "SELECT Id, Name FROM Account LIMIT 10"

        query = st.text_area(
            "SOQL Query:",
            value=default_query,
            height=150,
            help="Write your SOQL query here. Be careful with LIMIT clauses to avoid timeouts.",
        )

        if st.button("🚀 Execute Query", type="primary"):
            try:
                with st.spinner("Executing query..."):
                    result = st.session_state.sf_tool.run(
                        {"operation": "query", "query": query}
                    )
                    st.session_state.query_result = result
                    st.session_state.last_query = query

            except Exception as e:
                st.error(f"Query execution failed: {str(e)}")

    with col2:
        st.subheader("📚 SOQL Quick Reference")
        st.markdown(get_soql_reference())

    # Display results
    if "query_result" in st.session_state:
        result = st.session_state.query_result

        if isinstance(result, dict) and "records" in result:
            st.subheader("📊 Query Results")

            records_count = len(result["records"])
            total_size = result.get("totalSize", records_count)

            st.info(f"Retrieved {records_count} records (Total: {total_size})")

            if result["records"]:
                df = salesforce_to_dataframe(result)

                # Display as table
                st.dataframe(df, use_container_width=True)

                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"salesforce_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

                # Basic visualization options
                if len(df.columns) >= 2:
                    st.subheader("📈 Quick Visualization")

                    col1, col2 = st.columns(2)

                    with col1:
                        x_col = st.selectbox("X-axis:", df.columns)
                    with col2:
                        y_col = st.selectbox(
                            "Y-axis:", [col for col in df.columns if col != x_col]
                        )

                    if st.button("Create Chart"):
                        try:
                            if df[y_col].dtype in ["int64", "float64"]:
                                fig = px.bar(df, x=x_col, y=y_col)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                value_counts = df[x_col].value_counts()
                                fig = px.pie(
                                    values=value_counts.values,
                                    names=value_counts.index,
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Visualization error: {str(e)}")

            else:
                st.warning("Query returned no records")

        else:
            st.error("Query failed or returned unexpected format")
            st.json(result)


def main():
    """Main application function."""
    st.title("🚀 AI-Powered Salesforce Intelligence Platform")
    st.markdown("*Transform your CRM data into actionable business intelligence*")

    # Render sidebar and get selected page
    selected_page = render_sidebar()

    # Route to appropriate page - only Contact Intelligence available
    render_contact_intelligence()


if __name__ == "__main__":
    main()
