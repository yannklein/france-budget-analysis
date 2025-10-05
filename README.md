# 🏛️ Budget Horizon - French State Budget Analysis

A comprehensive Streamlit application for analyzing French government budget data with AI-powered predictions, inflation adjustment, and multi-language support.

## ✨ Features

- **📊 Interactive Visualizations**: Line charts, bar charts, stacked areas, and predictions
- **🌍 Multi-language Support**: Available in all 24 EU languages with dynamic translation
- **📈 AI Predictions**: Machine learning models for budget forecasting (2026-2030)
- **💰 Inflation Adjustment**: View amounts in constant euros with selectable base year
- **🏛️ Government Context**: Overlay government periods and key economic events
- **📚 Data Sources**: Transparent references to official government data
- **💳 Debt Analysis**: Includes state debt interest payments

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/BudgetHorizon.git
   cd BudgetHorizon
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501`

## 🌐 Deploy to Streamlit Cloud

### Option 1: Streamlit Community Cloud (Recommended)

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Select `app.py` as the main file
   - Click "Deploy!"

3. **Your app will be live** at `https://your-app-name.streamlit.app`

### Option 2: Other Platforms

**Railway:**
```bash
# Add railway.json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "streamlit run app.py --server.port $PORT"
  }
}
```

**Render:**
- Connect GitHub repo
- Select "Web Service"
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT`

## 📁 Project Structure

```
BudgetHorizon/
├── app.py                 # Main Streamlit application
├── data_fetcher.py        # Data fetching from government APIs
├── predictor.py          # ML models for budget predictions
├── utils.py              # Utility functions and i18n
├── requirements.txt      # Python dependencies
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── README.md             # This file
```

## 🔧 Configuration

### Environment Variables (Optional)
```bash
# For enhanced translation features
GOOGLE_TRANSLATE_API_KEY=your_key_here
```

### Streamlit Configuration
The app uses `.streamlit/config.toml` for:
- Server settings (headless mode, port)
- Theme customization
- CORS and security settings

## 📊 Data Sources

- **Budget Data**: [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/budget-de-letat/) (Ministry of Economy)
- **Inflation Data**: [INSEE](https://www.insee.fr/fr/statistiques/serie/000436391) (Consumer Price Index)
- **Debt Data**: [Banque de France](https://www.banque-france.fr/statistiques/dette-publique)

## 🛠️ Technical Details

- **Framework**: Streamlit 1.50+
- **Python**: 3.11+
- **ML Libraries**: scikit-learn, pandas, numpy
- **Visualization**: Plotly
- **Translation**: googletrans, deep-translator
- **Data Sources**: French government APIs with fallback data

## 🌍 Supported Languages

The app supports all 24 official EU languages:
- 🇫🇷 French (default)
- 🇬🇧 English
- 🇩🇪 German
- 🇪🇸 Spanish
- 🇮🇹 Italian
- 🇵🇱 Polish
- 🇷🇴 Romanian
- 🇳🇱 Dutch
- 🇬🇷 Greek
- 🇵🇹 Portuguese
- 🇨🇿 Czech
- 🇭🇺 Hungarian
- 🇸🇪 Swedish
- 🇧🇬 Bulgarian
- 🇦🇹 Austrian German
- 🇭🇷 Croatian
- 🇸🇰 Slovak
- 🇫🇮 Finnish
- 🇩🇰 Danish
- 🇱🇹 Lithuanian
- 🇸🇮 Slovenian
- 🇱🇻 Latvian
- 🇪🇪 Estonian
- 🇲🇹 Maltese

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- French government for open data initiatives
- Streamlit team for the amazing framework
- Plotly for interactive visualizations
- The open-source community

---

**Made with ❤️ for transparent government budget analysis**
