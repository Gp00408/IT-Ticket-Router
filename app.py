import streamlit as st

from route_ticket import TEAM_ROUTING, classify, generate_suggested_reply, load_artifacts

st.set_page_config(page_title="IT Ticket Router", page_icon="\U0001F3AB")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


st.title("IT Ticket Router")
st.write("Paste a support ticket below to see it classified and routed automatically.")

ticket_text = st.text_area(
    "Ticket text",
    height=120,
    placeholder="e.g. My laptop won't turn on and I have a meeting in an hour.",
)

if st.button("Route ticket", type="primary") and ticket_text.strip():
    vectorizer, category_model, urgency_model = get_artifacts()

    with st.spinner("Classifying..."):
        category, urgency = classify(
            ticket_text.strip(), vectorizer, category_model, urgency_model
        )
        team = TEAM_ROUTING.get(category, "General IT Support")
        suggested_reply = generate_suggested_reply(ticket_text.strip(), category, urgency)

    col1, col2 = st.columns(2)
    col1.metric("Category", category)
    col2.metric("Urgency", urgency)

    st.markdown(f"**Routed to:** {team}")
    st.markdown("**Suggested reply**")
    st.info(suggested_reply)
