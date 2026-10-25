import streamlit as st
import streamlit.components.v1 as components


def embed(html: str, height: int = 0):
    """Render a small HTML/JS snippet. Uses st.iframe on new Streamlit, components.html on older ones."""
    if hasattr(st, "iframe"):
        try:
            st.iframe(html, height=max(int(height), 1), width="stretch" if height else 1)
            return
        except Exception:
            pass
    components.html(html, height=height, width=0 if not height else None)
