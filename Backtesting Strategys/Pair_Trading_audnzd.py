import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm # Barra de progresso como na imagem

# Configuração visual "Como na imagem"
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (12, 6)

class PairTraderMCPT:
    def __init__(self, ticker_y, ticker_x, start_date='2018-01-01', end_date=None):
        self.ticker_y = ticker_y
        self.ticker_x = ticker_x
        self.data = self._fetch_data(start_date, end_date)
        
    def _fetch_data(self, start, end):
        print(f"📥 A descarregar dados: {self.ticker_y} vs {self.ticker_x}...")
        print(f"Start solicitado: {start}, End: {end}")
        
        try:
            df_raw = yf.download([self.ticker_y, self.ticker_x], start=start, end=end, auto_adjust=False, progress=False, group_by='ticker')
            
            # Extrair Close/Adj Close dependendo da disponibilidade
            # Usamos group_by='ticker' para facilitar a gestão de datas individuais
            def get_series(ticker):
                data_t = df_raw[ticker]
                return data_t['Adj Close'] if 'Adj Close' in data_t.columns else data_t['Close']

            series_y = get_series(self.ticker_y)
            series_x = get_series(self.ticker_x)

            # 1. Encontrar a 1ª data de cada um (removendo NaNs iniciais)
            first_y = series_y.dropna().index[0] if not series_y.dropna().empty else None
            first_x = series_x.dropna().index[0] if not series_x.dropna().empty else None

            print(f"📅 1ª data disponível {self.ticker_y}: {first_y.date() if first_y else 'N/A'}")
            print(f"📅 1ª data disponível {self.ticker_x}: {first_x.date() if first_x else 'N/A'}")

            if not first_y or not first_x:
                print("❌ Erro: Um dos tickers não retornou dados.")
                return pd.DataFrame()

            # 2. Primeira data comum (o máximo das duas primeiras datas)
            first_common = max(first_y, first_x)
            print(f"🚀 Estás a usar a 1ª data comum: {first_common.date()}")

            # Criar DataFrame final combinado a partir da data comum
            df = pd.DataFrame({
                'Y': series_y,
                'X': series_x
            })
            
            # Filtrar para começar na data comum e dropar NaNs restantes (buracos no meio ou no fim)
            df = df[df.index >= first_common].dropna()
            
            print(f"✅ Dados carregados. Período final: {df.index[0].date()} até {df.index[-1].date()} ({len(df)} linhas)")
            
            return df

        except Exception as e:
            print(f"Erro no download ou processamento: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def get_ols_hedge_ratio(self, series_y, series_x):
        """Calcula Beta via OLS (Otimização simples)"""
        X_add_c = sm.add_constant(series_x)
        model = sm.OLS(series_y, X_add_c).fit()
        return model.params.iloc[1]

    def run_strategy_engine(self, df_in, lookback_window=60):
        """
        Engine da estratégia corrigida para remover lookahead bias.
        Calcula o Beta (hedge ratio) e o Z-Score usando janelas móveis (rolling).
        """
        # 1. Rolling Hedge Ratio (Beta)
        # Usamos RollingOLS para calcular o beta em cada ponto t usando os últimos n dias.
        # IMPORTANTE: Fazemos .shift(1) para que no dia T usemos o beta calculado com dados até T-1.
        X_with_const = sm.add_constant(df_in['X'])
        rolling_model = RollingOLS(df_in['Y'], X_with_const, window=lookback_window).fit()
        hedge_ratio = rolling_model.params['X'].shift(1)
        
        # 2. Calcular Spread e Rolling Z-Score
        # spread_t = Y_t - beta_{t-1} * X_t
        # Nota: beta_{t-1} é o hedge_ratio já shiftado.
        spread = df_in['Y'] - hedge_ratio * df_in['X']
        
        # O Z-Score deve ser normalizado usando apenas dados passados.
        # Usamos a média e desvio padrão dos últimos 'lookback_window' spreads.
        # IMPORTANTE: Fazemos .shift(1) na média e std para evitar usar o spread de hoje no z-score de hoje.
        roll_mean = spread.rolling(window=lookback_window).mean().shift(1)
        roll_std = spread.rolling(window=lookback_window).std().shift(1)
        
        z_score = (spread - roll_mean) / roll_std
        
        # 3. Sinais (Threshold 2.0)
        # Long: spread muito baixo (abaixo de -2 desvios)
        long_pos = (z_score < -2.0).astype(float)
        # Fecha na média (z_score >= 0)
        long_pos = np.where(z_score >= 0, 0, long_pos)
        
        # Short: spread muito alto (acima de 2 desvios)
        short_pos = (z_score > 2.0).astype(float) * -1
        # Fecha na média (z_score <= 0)
        short_pos = np.where(z_score <= 0, 0, short_pos)
        
        # Combinação das posições
        # Como usamos .shift(1) nos parâmetros, o z_score de hoje já é 'limpo' de lookahead.
        # No entanto, a execução da ordem geralmente ocorre no fecho ou no dia seguinte.
        # Para ser conservador e realista, shiftamos as posições em 1 dia (sinal de hoje -> trade amanhã).
        positions = pd.Series(long_pos + short_pos, index=df_in.index).shift(1).fillna(0)
        
        # 4. Cálculo de Retornos do Par (PNL)
        # Retorno do par = retorno(Y) - beta * retorno(X)
        rets_y = df_in['Y'].pct_change().fillna(0)
        rets_x = df_in['X'].pct_change().fillna(0)
        
        # PNL = Posicao_{t-1} * (Retorno_Y_t - beta_{t-1} * Retorno_X_t)
        # Nota: 'hedge_ratio' já é o beta do período anterior.
        strategy_rets = positions * (rets_y - hedge_ratio.fillna(0) * rets_x)
        
        # Métrica de Performance (Sharpe Anualizado)
        if strategy_rets.std() == 0:
            return 0.0
        sharpe = (strategy_rets.mean() / strategy_rets.std()) * np.sqrt(252)
        return sharpe

    def get_permutation(self, df):
        """
        Quebra a correlação temporal embaralhando os retornos de X.
        Mantém a distribuição estatística de X, mas destrói a cointegração com Y.
        """
        df_perm = df.copy()
        
        # Calcular retornos
        rets_x = df['X'].pct_change().fillna(0).values
        
        # Embaralhar (Shuffle)
        np.random.shuffle(rets_x)
        
        # Reconstruir série de preços artificial de X
        # Começa no mesmo preço inicial para manter escala
        start_price = df['X'].iloc[0]
        price_path = start_price * (1 + rets_x).cumprod()
        
        df_perm['X'] = price_path
        return df_perm

    def run_mcpt(self, n_permutations=1000):
        """
        Monte Carlo Permutation Test (Lógica da Imagem)
        """
        print("\n🔎 A calcular Performance Real (Benchmark)...")
        # Performance Real
        best_real_pf = self.run_strategy_engine(self.data)
        print(f"Real Sharpe: {best_real_pf:.4f}")

        # Configuração do Teste (Igual à imagem)
        perm_better_count = 1  # Começa em 1 (Correção de viés / Add-one smoothing)
        permuted_pfs = []
        
        print(f"🎲 A rodar MCPT ({n_permutations} permutações)...")
        
        # Loop range(1, n) para contar o Real como a amostra 0
        for perm_i in tqdm(range(1, n_permutations)):
            
            # 1. Gerar dados permutados (como na imagem: train_perm = get_permutation)
            perm_df = self.get_permutation(self.data)
            
            # 2. "Otimizar" e rodar na permutação 
            # (Recalcula beta nos dados falsos para ver se acha padrão no ruído)
            best_perm_pf = self.run_strategy_engine(perm_df)
            
            # 3. Comparar
            if best_perm_pf >= best_real_pf:
                perm_better_count += 1
            
            permuted_pfs.append(best_perm_pf)

        # Cálculo do P-Value
        # (Casos melhores + 1) / (Total permutações)
        # Nota: Se n=1000 e range(1, 1000), temos 999 simulações + 1 real = 1000 total.
        p_value = perm_better_count / n_permutations
        
        print(f"\n📊 In-sample MCPT P-Value: {p_value:.5f}")
        
        self.plot_mcpt(permuted_pfs, best_real_pf, p_value)
        return p_value

    def plot_mcpt(self, permuted_pfs, real_pf, p_val):
        """Plota o histograma estilo Dark Mode igual à imagem"""
        
        # Criar figura com fundo escuro já definido pelo style
        fig, ax = plt.subplots()
        
        # Histograma dos Permutados
        sns.histplot(permuted_pfs, color='dodgerblue', kde=True, ax=ax, label='Permutations', stat='count')
        
        # Linha do Resultado Real
        ax.axvline(real_pf, color='red', linestyle='-', linewidth=3, label='Real')
        
        # Labels e Títulos
        ax.set_xlabel("Sharpe Ratio") # Ou Profit Factor
        ax.set_title(f"In-sample MCPT. P-Value: {p_val:.5f}", fontsize=14, color='white')
        ax.legend()
        
        # Ajustes de cor de texto para garantir leitura no fundo preto
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        
        plt.tight_layout()
        plt.show()

# --- Execução ---
if __name__ == "__main__":
    # 1. Inicializar
    trader = PairTraderMCPT('SI=F', 'SLV', start_date='1900-01-01')
    
    # 2. Rodar Teste de Permutação (MCPT)
    # Isto vai demorar alguns segundos devido ao loop 1000x
    trader.run_mcpt(n_permutations=1000)