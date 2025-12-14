import urllib.request
import urllib.error
import json
import sqlite3
from datetime import datetime, timedelta
import time
import ssl
import gzip
import os

# Configuração
NUM_QUARTERS = 80  # 20 anos (4 quarters por ano)
HEADERS = {
    'User-Agent': 'Individual Investor contact@gmail.com',  # Updated to look more realistic
    'Accept-Encoding': 'gzip, deflate',
    'Host': 'data.sec.gov'
}

# Caminho absoluto da base de dados fornecida
DB_PATH = r"c:\Users\Utilizador\Desktop\Documentosestudos\Prog\sec_financial_data_20251208_115157.db"

def make_request(url, is_sec_gov=True):
    """Faz requisição usando urllib com headers apropriados"""
    try:
        req = urllib.request.Request(url)
        for key, value in HEADERS.items():
            if key == 'Host' and not is_sec_gov:
                continue # Não enviar host data.sec.gov para www.sec.gov
            req.add_header(key, value)
        
        # SEC rate limit: max 10 requests per second. Let's be safe.
        if is_sec_gov:
            time.sleep(0.15)
        
        # Ignorar erros de certificado SSL (ambiente Windows/MinGW comum)
        context = ssl._create_unverified_context()
        
        with urllib.request.urlopen(req, context=context) as response:
            data = response.read()
            if response.info().get('Content-Encoding') == 'gzip':
                data = gzip.decompress(data)
            return json.loads(data)
    except urllib.error.HTTPError as e:
        print(f"Erro HTTP {e.code}: {e.reason} para {url}")
        return None
    except urllib.error.URLError as e:
        print(f"Erro de URL: {e.reason} para {url}")
        return None
    except Exception as e:
        print(f"Erro desconhecido: {e} para {url}")
        return None

def get_ticker_mapping():
    """Baixa o mapeamento de tickers da SEC"""
    print("Baixando mapeamento de tickers da SEC...")
    url = "https://www.sec.gov/files/company_tickers.json"
    
    try:
        req = urllib.request.Request(url)
        # Headers minimos para www.sec.gov
        req.add_header('User-Agent', HEADERS['User-Agent'])
        req.add_header('Accept-Encoding', 'gzip, deflate')
        
        # Ignorar erros de certificado SSL
        context = ssl._create_unverified_context()
        
        with urllib.request.urlopen(req, context=context) as response:
            data = response.read()
            if response.info().get('Content-Encoding') == 'gzip':
                data = gzip.decompress(data)
                
            json_data = json.loads(data)
            
            # O formato é { "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ... }
            mapping = {}
            for key, item in json_data.items():
                ticker = item['ticker'].upper()
                cik = str(item['cik_str']).zfill(10)
                mapping[ticker] = cik
            return mapping
            
    except Exception as e:
        print(f"Erro ao baixar mapeamento de tickers: {e}")
        return {}

def get_company_facts(cik):
    """Obtém os dados financeiros da empresa do SEC API"""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    return make_request(url, is_sec_gov=True)

def extract_quarterly_data(facts_data, metric_tags, units='USD'):
    """Extrai dados trimestrais tentando múltiplas tags"""
    try:
        if not facts_data or 'facts' not in facts_data or 'us-gaap' not in facts_data['facts']:
            return [], False, None

        us_gaap = facts_data['facts']['us-gaap']
        
        # Tenta cada tag na ordem fornecida
        for metric_tag in metric_tags:
            if metric_tag in us_gaap:
                metric_data = us_gaap[metric_tag]['units'].get(units, [])
                
                if metric_data:
                    # Filtra apenas dados trimestrais (10-Q) e anuais (10-K)
                    data = [
                        item for item in metric_data 
                        if item.get('form') in ['10-Q', '10-K']
                    ]
                    
                    if data:
                        # Ordena por data (mais recente primeiro)
                        data.sort(key=lambda x: (x.get('end', ''), x.get('filed', '')), reverse=True)
                        return data, True, metric_tag
        
        return [], False, None
    except (KeyError, TypeError):
        return [], False, None

def format_value(value):
    """Retorna valor numérico limpo"""
    if value is None:
        return None
    try:
        return float(value)
    except:
        return None

def get_db_connection():
    """Conecta à base de dados existente"""
    if not os.path.exists(DB_PATH):
        print(f"ERRO: Base de dados não encontrada em: {DB_PATH}")
        return None
        
    try:
        conn = sqlite3.connect(DB_PATH)
        return conn
    except sqlite3.Error as e:
        print(f"Erro ao conectar à base de dados: {e}")
        return None

def insert_data(conn, company_name, cik, fy, fp, end_date, filed, statement_type, 
                metric_name, metric_tag, value_type, value, units, existing_keys):
    """Insere dados na base de dados, verificando duplicatas via cache em memória"""
    
    # Chave única para verificação: (cik, fiscal_year, fiscal_period, metric_tag)
    # fiscal_year pode ser int ou str, normalizar para int se possivel ou str
    # O BD tem fiscal_year como INTEGER.
    
    key = (cik, int(fy) if str(fy).isdigit() else fy, fp, metric_tag)
    
    if key in existing_keys:
        # Já existe, pula
        return False

    cursor = conn.cursor()
    extraction_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Insere na tabela principal
    cursor.execute('''
        INSERT INTO financial_statements 
        (company_name, cik, fiscal_year, fiscal_period, end_date, filed_date, 
         statement_type, metric_name, metric_tag, value_type, value, units, extraction_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (company_name, cik, fy, fp, end_date, filed, statement_type, 
          metric_name, metric_tag, value_type, value, units, extraction_date))
    
    # Insere na tabela específica apenas se for valor trimestral (Q1, Q2, Q3, Q4)
    if value_type == 'Quarterly':
        table_map = {
            'Income Statement': 'income_statement',
            'Balance Sheet': 'balance_sheet',
            'Cash Flow Statement': 'cash_flow_statement'
        }
        
        table_name = table_map.get(statement_type)
        if table_name:
            cursor.execute(f'''
                INSERT INTO {table_name}
                (company_name, cik, fiscal_year, fiscal_period, end_date, metric_name, value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (company_name, cik, fy, fp, end_date, metric_name, value))
    
    conn.commit()
    # Atualiza o cache
    existing_keys.add(key)
    return True

def process_company(cik, conn):
    """Processa dados de uma empresa"""
    cursor = conn.cursor()
    
    # 1. Verifica se precisa de atualização
    cursor.execute("SELECT MAX(date(end_date)) FROM financial_statements WHERE cik = ?", (cik,))
    result = cursor.fetchone()
    last_date_str = result[0] if result else None
    
    should_update = False
    
    if not last_date_str:
        print(f"\nProcessando Novo CIK {cik}...")
        should_update = True
    else:
        try:
            # Tenta múltiplos formatos de data caso o SQL date() tenha falhado ou formato varie
            # O formato esperado é YYYY-MM-DD
            last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
            
            # Verifica se passaram 4 meses (aprox 120 dias)
            days_diff = (datetime.now() - last_date).days
            
            if days_diff >= 120:
                print(f"\nAtualizando CIK {cik} (Último registo: {last_date_str}, {days_diff} dias atrás)...")
                should_update = True
            else:
                print(f"  ✓ CIK {cik} está atualizado (Último: {last_date_str}).")
                return False # Não precisa atualizar
        except ValueError:
            print(f"\nData inválida encontrada para CIK {cik}: {last_date_str}. Forçando atualização...")
            should_update = True

    if not should_update:
        return False

    company_data = get_company_facts(cik)
    if not company_data:
        print(f"  ✗ Falha ao obter dados para CIK {cik}")
        return False
    
    company_name = company_data.get('entityName', 'Unknown')
    print(f"  Empresa: {company_name}")
    
    # Carregar chaves existentes para memória para verificação rápida de duplicatas
    # Obtém todas as combinações de (fiscal_year, fiscal_period, metric_tag) para este CIK
    cursor.execute("SELECT fiscal_year, fiscal_period, metric_tag FROM financial_statements WHERE cik = ?", (cik,))
    existing_rows = cursor.fetchall()
    existing_keys = set()
    for row in existing_rows:
        existing_keys.add((cik, row[0], row[1], row[2])) # tuple: (cik, fy, fp, tag)
    
    metrics = {
        'Income Statement': [
            (['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues','RevenuesNetOfInterestExpense','TotalNetSales','TotalRevenue'], 'Revenue', 'USD'),
            (['CostOfGoodsAndServicesSold', 'CostOfRevenue'], 'Cost of Goods and Services Sold', 'USD'),
            (['GrossProfit'], 'Gross Profit', 'USD'),
            (['ResearchAndDevelopmentExpense'], 'Research and Development', 'USD'),
            (['SellingGeneralAndAdministrativeExpense', 'GeneralAndAdministrativeExpense'], 'SG&A Expense', 'USD'),
            (['OperatingExpenses', 'OperatingCostsAndExpenses','NoninterestExpense','CostsAndExpenses'], 'Operating Expenses', 'USD'),
            (['OperatingIncomeLoss', 'IncomeLossFromOperations'], 'Operating Income', 'USD'),
            (['NonoperatingIncomeExpense', 'OtherNonoperatingIncomeExpense'], 'Nonoperating Income', 'USD'),
            (['IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest', 'IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments'], 'Pretax Income', 'USD'),
            (['IncomeTaxExpenseBenefit'], 'Income Tax Expense', 'USD'),
            (['NetIncomeLoss', 'ProfitLoss'], 'Net Income', 'USD'),
            (['EarningsPerShareBasic', 'EarningsPerShareDiluted'], 'EPS Basic', 'USD/shares'),
        ],
        'Balance Sheet': [
            (['CashAndCashEquivalentsAtCarryingValue', 'Cash'], 'Cash and Cash Equivalents', 'USD'),
            (['MarketableSecuritiesCurrent', 'ShortTermInvestments'], 'Marketable Securities - Current', 'USD'),
            (['AccountsReceivableNetCurrent', 'AccountsReceivableNet'], 'Accounts Receivable - Current', 'USD'),
            (['NontradeReceivablesCurrent', 'OtherReceivables'], 'Nontrade Receivables - Current', 'USD'),
            (['InventoryNet', 'Inventory'], 'Inventory', 'USD'),
            (['OtherAssetsCurrent', 'PrepaidExpenseAndOtherAssetsCurrent'], 'Other Current Assets', 'USD'),
            (['AssetsCurrent'], 'Total Current Assets', 'USD'),
            (['MarketableSecuritiesNoncurrent', 'LongTermInvestments'], 'Marketable Securities - Noncurrent', 'USD'),
            (['PropertyPlantAndEquipmentNet'], 'Property Plant and Equipment', 'USD'),
            (['OtherAssetsNoncurrent', 'OtherAssets'], 'Other Noncurrent Assets', 'USD'),
            (['AssetsNoncurrent'], 'Total Noncurrent Assets', 'USD'),
            (['Assets'], 'Total Assets', 'USD'),
            (['AccountsPayableCurrent', 'AccountsPayable'], 'Accounts Payable', 'USD'),
            (['OtherLiabilitiesCurrent', 'AccruedLiabilitiesCurrent'], 'Other Current Liabilities', 'USD'),
            (['ContractWithCustomerLiabilityCurrent', 'DeferredRevenueCurrent'], 'Deferred Revenue', 'USD'),
            (['CommercialPaper'], 'Commercial Paper', 'USD'),
            (['LongTermDebtCurrent', 'LongTermDebtAndCapitalLeaseObligationsCurrent'], 'Long-Term Debt - Current', 'USD'),
            (['LiabilitiesCurrent'], 'Total Current Liabilities', 'USD'),
            (['LongTermDebtNoncurrent', 'LongTermDebtAndCapitalLeaseObligations'], 'Long-Term Debt - Noncurrent', 'USD'),
            (['OtherLiabilitiesNoncurrent'], 'Other Noncurrent Liabilities', 'USD'),
            (['LiabilitiesNoncurrent'], 'Total Noncurrent Liabilities', 'USD'),
            (['Liabilities'], 'Total Liabilities', 'USD'),
        ],
        'Cash Flow Statement': [
            (['CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents', 'CashAndCashEquivalentsAtCarryingValue'], 'Cash Beginning', 'USD'),
            (['NetIncomeLoss', 'ProfitLoss'], 'Net Income', 'USD'),
            (['DepreciationDepletionAndAmortization', 'Depreciation'], 'Depreciation and Amortization', 'USD'),
            (['ShareBasedCompensation', 'AllocatedShareBasedCompensationExpense'], 'Share-Based Compensation', 'USD'),
            (['OtherNoncashIncomeExpense'], 'Other Noncash Items', 'USD'),
            (['IncreaseDecreaseInAccountsReceivable'], 'Change in Accounts Receivable', 'USD'),
            (['IncreaseDecreaseInOtherReceivables'], 'Change in Other Receivables', 'USD'),
            (['IncreaseDecreaseInInventories'], 'Change in Inventories', 'USD'),
            (['IncreaseDecreaseInOtherOperatingAssets'], 'Change in Other Operating Assets', 'USD'),
            (['IncreaseDecreaseInAccountsPayable'], 'Change in Accounts Payable', 'USD'),
            (['IncreaseDecreaseInOtherOperatingLiabilities'], 'Change in Other Operating Liabilities', 'USD'),
            (['NetCashProvidedByUsedInOperatingActivities'], 'Operating Cash Flow', 'USD'),
            (['PaymentsToAcquireAvailableForSaleSecuritiesDebt', 'PaymentsToAcquireInvestments'], 'Purchase of Securities', 'USD'),
            (['ProceedsFromMaturitiesPrepaymentsAndCallsOfAvailableForSaleSecurities', 'ProceedsFromSaleMaturityAndCollectionsOfInvestments'], 'Proceeds from Maturities', 'USD'),
            (['ProceedsFromSaleOfAvailableForSaleSecuritiesDebt', 'ProceedsFromSaleOfInvestments'], 'Proceeds from Sales of Securities', 'USD'),
            (['PaymentsToAcquirePropertyPlantAndEquipment', 'CapitalExpendituresIncurredButNotYetPaid'], 'Capital Expenditures', 'USD'),
            (['PaymentsForProceedsFromOtherInvestingActivities'], 'Other Investing Activities', 'USD'),
            (['NetCashProvidedByUsedInInvestingActivities'], 'Investing Cash Flow', 'USD'),
            (['PaymentsRelatedToTaxWithholdingForShareBasedCompensation'], 'Tax Withholding', 'USD'),
            (['PaymentsOfDividends', 'PaymentsOfDividendsCommonStock'], 'Dividends Paid', 'USD'),
            (['PaymentsForRepurchaseOfCommonStock', 'TreasuryStockValueAcquiredCostMethod'], 'Stock Repurchases', 'USD'),
            (['ProceedsFromIssuanceOfLongTermDebt', 'ProceedsFromDebtNetOfIssuanceCosts'], 'Proceeds from Debt', 'USD'),
            (['RepaymentsOfLongTermDebt', 'RepaymentsOfDebt'], 'Debt Repayments', 'USD'),
            (['ProceedsFromRepaymentsOfCommercialPaper'], 'Change in Commercial Paper', 'USD'),
            (['ProceedsFromPaymentsForOtherFinancingActivities'], 'Other Financing Activities', 'USD'),
            (['NetCashProvidedByUsedInFinancingActivities'], 'Financing Cash Flow', 'USD'),
            (['CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect', 'CashAndCashEquivalentsPeriodIncreaseDecrease'], 'Net Change in Cash', 'USD'),
        ]
    }
    
    total_inserted = 0
    
    for statement_type, statement_metrics in metrics.items():
        for metric_tags, metric_label, units in statement_metrics:
            all_data, found, used_tag = extract_quarterly_data(company_data, metric_tags, units)
            
            if found and all_data:
                fiscal_years = {}
                
                for item in all_data:
                    fy = item.get('fy')
                    fp = item.get('fp', 'FY')
                    end_date = item.get('end')
                    value = item.get('val')
                    filed = item.get('filed')
                    form = item.get('form')
                    
                    if not fy or not end_date:
                        continue
                    
                    if fy not in fiscal_years:
                        fiscal_years[fy] = {}
                    
                    if form == '10-K':
                        key = 'Full Year'
                    else:
                        key = fp if fp != 'FY' else 'Full Year'
                    
                    if key not in fiscal_years[fy] or filed > fiscal_years[fy][key].get('filed', ''):
                        fiscal_years[fy][key] = {
                            'value': value,
                            'end_date': end_date,
                            'filed': filed,
                            'fp': fp,
                            'form': form
                        }
                
                sorted_years = sorted(fiscal_years.keys(), reverse=True)
                quarters_shown = 0
                ytd_shown = False
                
                for fy in sorted_years:
                    if quarters_shown >= NUM_QUARTERS:
                        break
                    
                    year_data = fiscal_years[fy]
                    
                    # YTD para o ano mais recente
                    if not ytd_shown:
                        latest_q = None
                        for q in ['Q4', 'Q3', 'Q2', 'Q1']:
                            if q in year_data:
                                latest_q = q
                                break
                        
                        if latest_q:
                            q_data = year_data[latest_q]
                            success = insert_data(conn, company_name, cik, fy, latest_q, 
                                      q_data['end_date'], q_data['filed'], statement_type,
                                      metric_label, used_tag, 'YTD', 
                                      format_value(q_data['value']), units, existing_keys)
                            if success: total_inserted += 1
                            ytd_shown = True
                    
                    # Criar Q4 artificialmente se tivermos Full Year e os 3 quarters
                    if 'Full Year' in year_data and quarters_shown < NUM_QUARTERS:
                        fy_data = year_data['Full Year']
                        fy_value = fy_data['value']
                        
                        has_q1 = 'Q1' in year_data
                        has_q2 = 'Q2' in year_data
                        has_q3 = 'Q3' in year_data
                        
                        if has_q1 and has_q2 and has_q3 and fy_value is not None:
                            q1_ytd = year_data['Q1']['value'] or 0
                            q2_ytd = year_data['Q2']['value'] or 0
                            q3_ytd = year_data['Q3']['value'] or 0
                            
                            q1_quarterly = q1_ytd
                            q2_quarterly = q2_ytd - q1_ytd
                            q3_quarterly = q3_ytd - q2_ytd
                            
                            q4_value = fy_value - q1_quarterly - q2_quarterly - q3_quarterly
                            
                            success = insert_data(conn, company_name, cik, fy, 'Q4',
                                      fy_data['end_date'], fy_data['filed'], statement_type,
                                      metric_label, used_tag, 'Quarterly',
                                      format_value(q4_value), units, existing_keys)
                            if success: total_inserted += 1
                            quarters_shown += 1
                    
                    # Quarters individuais Q3, Q2, Q1
                    for q in ['Q3', 'Q2', 'Q1']:
                        if quarters_shown >= NUM_QUARTERS:
                            break
                        
                        if q in year_data:
                            q_data = year_data[q]
                            quarterly_value = q_data['value']
                            
                            prev_q = {'Q3': 'Q2', 'Q2': 'Q1', 'Q1': None}[q]
                            if prev_q and prev_q in year_data:
                                prev_value = year_data[prev_q]['value']
                                if prev_value and quarterly_value:
                                    quarterly_value = quarterly_value - prev_value
                            
                            success = insert_data(conn, company_name, cik, fy, q,
                                      q_data['end_date'], q_data['filed'], statement_type,
                                      metric_label, used_tag, 'Quarterly',
                                      format_value(quarterly_value), units, existing_keys)
                            if success: total_inserted += 1
                            quarters_shown += 1
                    
                    # Full Year completo (guarda na tabela principal apenas)
                    if 'Full Year' in year_data:
                        fy_data = year_data['Full Year']
                        success = insert_data(conn, company_name, cik, fy, 'FY',
                                  fy_data['end_date'], fy_data['filed'], statement_type,
                                  metric_label, used_tag, 'Full Year',
                                  format_value(fy_data['value']), units, existing_keys)
                        if success: total_inserted += 1
    
    print(f"  ✓ {total_inserted} NOVOS registos inseridos")
    return True

def main():
    print("Iniciando processo de ATUALIZAÇÃO de dados por Ticker...")
    print(f"Base de Dados Alvo: {DB_PATH}")
    
    # 1. Conectar à base de dados
    conn = get_db_connection()
    if not conn:
        print("Impossível prosseguir sem base de dados.")
        return

    # 2. Ler Tickers do arquivo
    try:
        with open('Tickers.txt', 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
    except FileNotFoundError:
        print("Erro: Ficheiro Tickers.txt não encontrado!")
        return
    
    if not tickers:
        print("Erro: Ficheiro Tickers.txt está vazio!")
        conn.close()
        return
        
    print(f"Lendo {len(tickers)} Tickers para verificação...")
    
    # 3. Obter mapa de CIKs
    ticker_map = get_ticker_mapping()
    if not ticker_map:
        print("Falha download do mapa de tickers.")
        conn.close()
        return 
    
    ciks_map = {}
    for ticker in tickers:
        if ticker in ticker_map:
            ciks_map[ticker] = ticker_map[ticker]
        else:
            print(f"  Aviso: CIK não encontrado para {ticker}")
            
    # 4. Processar cada empresa
    print(f"\nVerificando {len(ciks_map)} empresas por atualizações...")
    
    processed_count = 0
    for ticker, cik in ciks_map.items():
        if process_company(cik, conn):
            processed_count += 1
    
    conn.close()
    
    print(f"\n{'='*80}")
    print(f"ATUALIZAÇÃO CONCLUÍDA")
    print(f"{'='*80}")
    print(f"Empresas atualizadas/processadas: {processed_count}")
    print(f"Total verificado: {len(ciks_map)}")

if __name__ == "__main__":
    main()
