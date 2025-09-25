import dash
from dash import html, dcc

app = dash.Dash(__name__)

app.layout = html.Div([

    # Fixed header
    html.Div(
        "📌 Fixed Header (always visible)",
        style={
            "position": "fixed",   # ✅ Fix relative to viewport
            "top": "0",
            "left": "0",
            "right": "0",
            "height": "60px",
            "background": "#f8f9fa",
            "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
            "zIndex": "1000",      # ✅ ensure it sits above other content
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "fontWeight": "bold",
            "fontSize": "20px"
        }
    ),

    # Scrollable content
    html.Div(
        children=[
            html.H2("Scrollable Content Section"),
            html.P("This section scrolls while the header stays fixed.")]*50,
        style={
            "marginTop": "70px",   # ✅ leave space for header
            "padding": "20px"
        }
    )
])

if __name__ == "__main__":
    app.run(debug=True)
