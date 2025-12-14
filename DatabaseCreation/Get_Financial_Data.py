import requests
import sqlite3
from datetime import datetime

# Configuração
NUM_QUARTERS = 40  # Número de quarters a mostrar
HEADERS = {
    'User-Agent': 'SeuNome seu@email.com',  # IMPORTANTE: Substituir com seu email
    'Accept-Encoding': 'gzip, deflate',
    'Host': 'data.sec.gov'
}

def get_company_facts(cik):
    """Obtém os dados financeiros da empresa do SEC API"""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Erro ao obter dados: {e}")
        return None

def extract_quarterly_data(facts_data, metric_tags, units='USD'):
    """Extrai dados trimestrais tentando múltiplas tags"""
    try:
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

def create_database(db_name):
    """Cria a base de dados e as tabelas"""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Tabela principal com todos os dados
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS financial_statements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT,
            cik TEXT,
            fiscal_year INTEGER,
            fiscal_period TEXT,
            end_date TEXT,
            filed_date TEXT,
            statement_type TEXT,
            metric_name TEXT,
            metric_tag TEXT,
            value_type TEXT,
            value REAL,
            units TEXT,
            extraction_date TEXT
        )
    ''')
    
    # Tabela Income Statement
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS income_statement (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT,
            cik TEXT,
            fiscal_year INTEGER,
            fiscal_period TEXT,
            end_date TEXT,
            metric_name TEXT,
            value REAL
        )
    ''')
    
    # Tabela Balance Sheet
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS balance_sheet (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT,
            cik TEXT,
            fiscal_year INTEGER,
            fiscal_period TEXT,
            end_date TEXT,
            metric_name TEXT,
            value REAL
        )
    ''')
    
    # Tabela Cash Flow Statement
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cash_flow_statement (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT,
            cik TEXT,
            fiscal_year INTEGER,
            fiscal_period TEXT,
            end_date TEXT,
            metric_name TEXT,
            value REAL
        )
    ''')
    
    conn.commit()
    return conn

def insert_data(conn, company_name, cik, fy, fp, end_date, filed, statement_type, 
                metric_name, metric_tag, value_type, value, units):
    """Insere dados na base de dados"""
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

def process_company(cik, conn):
    """Processa dados de uma empresa"""
    print(f"\nProcessando CIK {cik}...")
    
    company_data = get_company_facts(cik)
    if not company_data:
        print(f"  ✗ Falha ao obter dados para CIK {cik}")
        return False
    
    company_name = company_data.get('entityName', 'Unknown')
    print(f"  Empresa: {company_name}")
    
    # Métricas organizadas por demonstração financeira (com tags alternativas)
    metrics = {
        'Income Statement': [
            (['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues'], 'Revenue', 'USD'),
            (['CostOfGoodsAndServicesSold', 'CostOfRevenue'], 'Cost of Goods and Services Sold', 'USD'),
            (['GrossProfit'], 'Gross Profit', 'USD'),
            (['ResearchAndDevelopmentExpense'], 'Research and Development', 'USD'),
            (['SellingGeneralAndAdministrativeExpense', 'GeneralAndAdministrativeExpense'], 'SG&A Expense', 'USD'),
            (['OperatingExpenses', 'OperatingCostsAndExpenses'], 'Operating Expenses', 'USD'),
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
                    
                    # Se for 10-K, é Full Year
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
                            insert_data(conn, company_name, cik, fy, latest_q, 
                                      q_data['end_date'], q_data['filed'], statement_type,
                                      metric_label, used_tag, 'YTD', 
                                      format_value(q_data['value']), units)
                            total_inserted += 1
                            ytd_shown = True
                    
                    # Criar Q4 artificialmente se tivermos Full Year e os 3 quarters
                    if 'Full Year' in year_data and quarters_shown < NUM_QUARTERS:
                        fy_data = year_data['Full Year']
                        fy_value = fy_data['value']
                        
                        # Verifica se temos Q1, Q2, Q3
                        has_q1 = 'Q1' in year_data
                        has_q2 = 'Q2' in year_data
                        has_q3 = 'Q3' in year_data
                        
                        if has_q1 and has_q2 and has_q3 and fy_value is not None:
                            # Obter valores YTD da API
                            q1_ytd = year_data['Q1']['value'] or 0
                            q2_ytd = year_data['Q2']['value'] or 0
                            q3_ytd = year_data['Q3']['value'] or 0
                            
                            # Converter YTD para valores trimestrais
                            q1_quarterly = q1_ytd
                            q2_quarterly = q2_ytd - q1_ytd
                            q3_quarterly = q3_ytd - q2_ytd
                            
                            # Calcula Q4 = Full Year - (Q1 + Q2 + Q3)
                            q4_value = fy_value - q1_quarterly - q2_quarterly - q3_quarterly
                            
                            # Insere Q4 calculado
                            insert_data(conn, company_name, cik, fy, 'Q4',
                                      fy_data['end_date'], fy_data['filed'], statement_type,
                                      metric_label, used_tag, 'Quarterly',
                                      format_value(q4_value), units)
                            total_inserted += 1
                            quarters_shown += 1
                    
                    # Quarters individuais Q3, Q2, Q1
                    for q in ['Q3', 'Q2', 'Q1']:
                        if quarters_shown >= NUM_QUARTERS:
                            break
                        
                        if q in year_data:
                            q_data = year_data[q]
                            quarterly_value = q_data['value']
                            
                            # Lógica para Q1, Q2, Q3
                            prev_q = {'Q3': 'Q2', 'Q2': 'Q1', 'Q1': None}[q]
                            if prev_q and prev_q in year_data:
                                prev_value = year_data[prev_q]['value']
                                if prev_value and quarterly_value:
                                    quarterly_value = quarterly_value - prev_value
                            
                            insert_data(conn, company_name, cik, fy, q,
                                      q_data['end_date'], q_data['filed'], statement_type,
                                      metric_label, used_tag, 'Quarterly',
                                      format_value(quarterly_value), units)
                            total_inserted += 1
                            quarters_shown += 1
                    
                    # Full Year completo (guarda na tabela principal apenas)
                    if 'Full Year' in year_data:
                        fy_data = year_data['Full Year']
                        insert_data(conn, company_name, cik, fy, 'FY',
                                  fy_data['end_date'], fy_data['filed'], statement_type,
                                  metric_label, used_tag, 'Full Year',
                                  format_value(fy_data['value']), units)
                        total_inserted += 1
    
    print(f"  ✓ {total_inserted} registos inseridos")
    return True

def main():
    # Lê CIKs do ficheiro
    try:
        with open('CIK.txt', 'r') as f:
            ciks = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Erro: Ficheiro CIK.txt não encontrado!")
        print("Cria um ficheiro CIK.txt com um CIK por linha (ex: 0000320193)")
        return
    
    if not ciks:
        print("Erro: Ficheiro CIK.txt está vazio!")
        return
    
    print(f"Encontrados {len(ciks)} CIK(s) para processar")
    
    # Cria base de dados
    db_name = f"sec_financial_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    conn = create_database(db_name)
    print(f"\nBase de dados criada: {db_name}")
    
    # Processa cada empresa
    successful = 0
    for cik in ciks:
        if process_company(cik, conn):
            successful += 1
    
    conn.close()
    
    print(f"\n{'='*80}")
    print(f"PROCESSAMENTO CONCLUÍDO")
    print(f"{'='*80}")
    print(f"Base de dados: {db_name}")
    print(f"Empresas processadas: {successful}/{len(ciks)}")
    print(f"\nTabelas criadas:")
    print(f"  1. financial_statements (tabela principal com todos os dados)")
    print(f"  2. income_statement (apenas Income Statement - valores trimestrais)")
    print(f"  3. balance_sheet (apenas Balance Sheet - valores trimestrais)")
    print(f"  4. cash_flow_statement (apenas Cash Flow - valores trimestrais)")

if __name__ == "__main__":
    main()