"""
股票分析助手 - 版本 6
新增功能：Prophet周期性分析工具
"""
import json
import os
import decimal
from typing import List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import plotly.utils
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import time

from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


@register_tool('ExcSql')
class ExcSql(BaseTool):
    description = '执行SQL查询并自动生成图表'
    parameters = [
        {
            'name': 'sql',
            'type': 'string',
            'description': '要执行的SQL查询语句',
            'required': True
        }
    ]

    def call(self, params: str, **kwargs) -> List[Dict[str, Any]]:
        """
        执行SQL查询
        :param params: JSON字符串，包含sql参数
        :param kwargs: 其他参数（如messages等）
        :return: 查询结果
        """
        import sqlite3
        params_dict = json.loads(params)
        sql = params_dict.get('sql')

        try:
            # 连接本地SQLite数据库
            connection = sqlite3.connect('stock_data.db')
            
            cursor = connection.cursor()
            
            # 由于数据库中的日期现在是标准格式（YYYY-MM-DD），我们处理日期函数使其适用于标准日期格式
            processed_sql = self._process_sql_for_standard_date(sql)
            
            cursor.execute(processed_sql)
            result = cursor.fetchall()
            
            # 获取列名
            columns = [desc[0] for desc in cursor.description]
            
            # 将结果转换为字典列表
            rows = []
            for row in result:
                row_dict = {}
                for i, col in enumerate(columns):
                    value = row[i]
                    # 将Decimal类型转换为float，以便JSON序列化
                    if isinstance(value, decimal.Decimal):
                        value = float(value)
                    row_dict[col] = value
                rows.append(row_dict)
            
            # 创建DataFrame并生成图表
            if rows:
                df = pd.DataFrame(rows)
                
                # 自动创建目录
                save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
                os.makedirs(save_dir, exist_ok=True)
                filename = f'chart_{int(time.time() * 1000)}.png'
                save_path = os.path.join(save_dir, filename)
                
                # 生成图表
                generate_chart_png(df, save_path)
                
                # 转换为Markdown格式的图片链接
                img_path = os.path.join('image_show', filename)
                img_md = f'![股票数据图表]({img_path})'
                
                # 如果结果数量过多（超过100行），只返回汇总信息而不是全部数据
                if len(rows) > 100:
                    # 为大量数据创建汇总表格
                    summary_data = {
                        '总记录数': len(rows),
                    }
                    
                    # 根据实际列名进行汇总统计
                    if 'trade_date' in df.columns:
                        summary_data['日期范围'] = f"{df['trade_date'].min()} 至 {df['trade_date'].max()}"
                    
                    # 数值列的统计
                    numeric_columns = ['open', 'high', 'low', 'close', 'pct_chg', 'change', 'vol', 'amount']
                    for col in numeric_columns:
                        if col in df.columns:
                            try:
                                if col in ['open', 'high', 'low', 'close']:
                                    summary_data[f'{col.capitalize()}价范围'] = f"{df[col].min():.2f} - {df[col].max():.2f}"
                                elif col == 'pct_chg':
                                    summary_data['涨跌幅范围'] = f"{df[col].min():.2f}% - {df[col].max():.2f}%"
                                    summary_data['平均涨跌幅'] = f"{df[col].mean():.2f}%"
                                elif col == 'change':
                                    summary_data['涨跌额范围'] = f"{df[col].min():.2f} - {df[col].max():.2f}"
                                elif col in ['vol', 'amount']:
                                    summary_data[f'{col.capitalize()}范围'] = f"{df[col].min():.0f} - {df[col].max():.0f}"
                            except:
                                summary_data[f'{col.capitalize()}信息'] = 'N/A'
                    
                    # 如果包含股票代码或名称，也进行统计
                    if 'ts_code' in df.columns:
                        unique_codes = df['ts_code'].nunique()
                        summary_data['不同股票数'] = unique_codes
                        if unique_codes <= 5:  # 如果股票种类不多，列出所有股票
                            codes_list = ', '.join(df['ts_code'].unique())
                            summary_data['股票代码'] = codes_list
                    if 'stock_name' in df.columns:
                        unique_names = df['stock_name'].nunique()
                        if unique_names <= 5:  # 如果股票种类不多，列出所有股票名
                            names_list = ', '.join(df['stock_name'].unique())
                            summary_data['股票名称'] = names_list
                    
                    # 创建汇总DataFrame
                    summary_df = pd.DataFrame([summary_data])
                    md = summary_df.to_markdown(index=False) if hasattr(summary_df, 'to_markdown') else str(summary_df)
                    
                    # 添加说明文字
                    md = f"数据量较大（{len(rows)} 条记录），以下是汇总信息：\n\n{md}\n\n完整数据已生成图表，详情请查看下方图表。"
                    
                    return [{'result': rows, 'count': len(rows), 'table': md, 'chart': img_md}]
                else:
                    # 如果数据量适中，返回完整数据
                    md = df.to_markdown(index=False) if hasattr(df, 'to_markdown') else str(df)
                    
                    return [{'result': rows, 'count': len(rows), 'table': md, 'chart': img_md}]
            else:
                md = pd.DataFrame(rows).to_markdown(index=False) if hasattr(pd.DataFrame(rows), 'to_markdown') else str(pd.DataFrame(rows))
                return [{'result': rows, 'count': len(rows), 'table': md}]
            
        except Exception as e:
            return [{'error': f'执行SQL查询时出错: {str(e)}'}]
        finally:
            if 'connection' in locals():
                connection.close()

    def _process_sql_for_standard_date(self, sql: str) -> str:
        """
        处理SQL语句，将标准日期函数转换为适用于标准日期格式（YYYY-MM-DD）的函数
        SQLite的DATE函数可以直接使用，因为我们现在使用标准日期格式
        """
        import re
        from datetime import datetime, timedelta
        import calendar
        
        # 辅助函数：根据偏移量计算目标日期
        def get_target_date_from_offset(offset_str):
            now = datetime.now()
            
            # 解析不同的时间偏移表达式
            if "'-1 month'" in offset_str or '"-1 month"' in offset_str:
                # 计算一个月前的日期
                if now.month == 1:
                    target_year = now.year - 1
                    target_month = 12
                else:
                    target_year = now.year
                    target_month = now.month - 1
                
                # 获取目标月份的天数，避免日期超出范围
                _, last_day = calendar.monthrange(target_year, target_month)
                target_day = min(now.day, last_day)
                
                target_date = datetime(target_year, target_month, target_day)
                return target_date.strftime('%Y-%m-%d')
            elif "'-2 month'" in offset_str or '"-2 month"' in offset_str:
                # 计算两个月前的日期
                target_month = now.month - 2
                target_year = now.year
                while target_month <= 0:
                    target_month += 12
                    target_year -= 1
                
                # 获取目标月份的天数
                _, last_day = calendar.monthrange(target_year, target_month)
                target_day = min(now.day, last_day)
                
                target_date = datetime(target_year, target_month, target_day)
                return target_date.strftime('%Y-%m-%d')
            elif "'-3 month'" in offset_str or '"-3 month"' in offset_str:
                # 计算三个月前的日期
                target_month = now.month - 3
                target_year = now.year
                while target_month <= 0:
                    target_month += 12
                    target_year -= 1
                
                # 获取目标月份的天数
                _, last_day = calendar.monthrange(target_year, target_month)
                target_day = min(now.day, last_day)
                
                target_date = datetime(target_year, target_month, target_day)
                return target_date.strftime('%Y-%m-%d')
            elif "'-6 month'" in offset_str or '"-6 month"' in offset_str:
                # 计算六个月前的日期
                target_month = now.month - 6
                target_year = now.year
                while target_month <= 0:
                    target_month += 12
                    target_year -= 1
                
                # 获取目标月份的天数
                _, last_day = calendar.monthrange(target_year, target_month)
                target_day = min(now.day, last_day)
                
                target_date = datetime(target_year, target_month, target_day)
                return target_date.strftime('%Y-%m-%d')
            elif "'-1 year'" in offset_str or '"-1 year"' in offset_str:
                # 计算一年前的日期
                target_date = datetime(now.year - 1, now.month, now.day)
                return target_date.strftime('%Y-%m-%d')
            elif "'+1 day'" in offset_str or '"+1 day"' in offset_str:
                # 计算一天后的日期
                target_date = now + timedelta(days=1)
                return target_date.strftime('%Y-%m-%d')
            elif "'-1 day'" in offset_str or '"-1 day"' in offset_str:
                # 计算一天前的日期
                target_date = now - timedelta(days=1)
                return target_date.strftime('%Y-%m-%d')
            elif "'now'" in offset_str or '"now"' in offset_str:
                # 返回当前日期
                return now.strftime('%Y-%m-%d')
            else:
                # 尝试解析动态偏移量
                # 匹配如 '-30 days', '+7 days', '-3 months' 等
                dynamic_pattern = r"([+-]\d+)\s*(day|days|week|weeks|month|months|year|years)"
                match = re.search(dynamic_pattern, offset_str.lower())
                
                if match:
                    amount = int(match.group(1))
                    unit = match.group(2)
                    
                    if 'month' in unit:
                        # 处理月份偏移
                        target_month = now.month + amount
                        target_year = now.year
                        while target_month > 12:
                            target_month -= 12
                            target_year += 1
                        while target_month <= 0:
                            target_month += 12
                            target_year -= 1
                        
                        # 获取目标月份的天数
                        _, last_day = calendar.monthrange(target_year, target_month)
                        target_day = min(now.day, last_day)
                        
                        target_date = datetime(target_year, target_month, target_day)
                        return target_date.strftime('%Y-%m-%d')
                    elif 'week' in unit:
                        # 处理周偏移
                        target_date = now + timedelta(weeks=amount)
                        return target_date.strftime('%Y-%m-%d')
                    elif 'year' in unit:
                        # 处理年偏移
                        target_date = datetime(now.year + amount, now.month, now.day)
                        return target_date.strftime('%Y-%m-%d')
                    else:  # days
                        # 处理天偏移
                        target_date = now + timedelta(days=amount)
                        return target_date.strftime('%Y-%m-%d')
                        
            # 如果无法解析，则返回当前日期
            return now.strftime('%Y-%m-%d')
        
        # 替换条件中的日期函数
        def replace_condition_date_function(match):
            comparison_op = match.group(1)
            date_part = match.group(2)
            target_date = get_target_date_from_offset(date_part)
            return f"trade_date {comparison_op} '{target_date}'"
        
        # 替换独立的日期函数
        def replace_standalone_date_function(match):
            date_part = match.group(1)
            target_date = get_target_date_from_offset(date_part)
            return f"'{target_date}'"
        
        # 处理带比较运算符的日期函数 (如: trade_date >= DATE('now', '-1 month'))
        sql = re.sub(r"trade_date\s*(>=|<=|>|<|!=|=)\s*DATE\s*\(\s*(['\"].*)\)", 
                     replace_condition_date_function, sql, flags=re.IGNORECASE)
        
        # 处理其他上下文中的DATE函数
        # 注意：这里要小心，避免替换可能不是日期函数的情况
        sql = re.sub(r"DATE\s*\(\s*(['\"].*)\)", 
                     replace_standalone_date_function, sql, flags=re.IGNORECASE)
        
        return sql


def generate_chart_png(df_sql, save_path):
    """
    生成股票收盘价折线图
    """
    # 检查是否有交易日期列和收盘价列
    date_columns = [col for col in df_sql.columns if 'date' in col.lower()]
    close_column = [col for col in df_sql.columns if 'close' in col.lower()]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if date_columns and close_column:
        # 使用交易日期作为X轴，收盘价作为Y轴
        date_col = date_columns[0]
        close_col = close_column[0]
        x_values = df_sql[date_col].apply(str)  # 转换为字符串便于显示
        
        ax.plot(x_values, df_sql[close_col], marker='o', label=f'{close_col}价格', linewidth=2, color='#1f77b4')
        ax.set_xlabel(date_col)
        ax.set_ylabel(close_col)
        ax.set_title('股票收盘价走势图')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 优化日期标签显示 - 如果数据点太多，只显示部分标签
        if len(x_values) > 10:  # 如果数据点超过10个
            # 计算要显示的标签索引，大约每10个点显示一个标签，最多显示10个标签
            step = max(1, len(x_values) // 10)  # 每10个点显示一个
            tick_indices = list(range(0, len(x_values), step))
            tick_labels = [x_values.iloc[i] for i in tick_indices]
            
            # 设置X轴刻度和标签
            ax.set_xticks(tick_indices)
            ax.set_xticklabels(tick_labels, rotation=45)
        else:
            # 如果数据点不多，显示所有标签但旋转45度
            ax.tick_params(axis='x', rotation=45)
            
    elif close_column:
        # 如果没有日期列，使用索引作为X轴
        close_col = close_column[0]
        ax.plot(df_sql[close_col], marker='o', label=f'{close_col}价格', linewidth=2, color='#1f77b4')
        ax.set_xlabel('Index')
        ax.set_ylabel(close_col)
        ax.set_title('股票收盘价走势图')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 如果数据点太多，优化X轴标签显示
        if len(df_sql) > 10:
            step = max(1, len(df_sql) // 10)
            tick_indices = list(range(0, len(df_sql), step))
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([str(i) for i in tick_indices])
    else:
        # 如果没有收盘价列，尝试使用第一个数值列
        numeric_columns = df_sql.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            first_numeric = numeric_columns[0]
            if date_columns:
                date_col = date_columns[0]
                x_values = df_sql[date_col].apply(str)
                ax.plot(x_values, df_sql[first_numeric], marker='o', label=first_numeric, linewidth=2, color='#1f77b4')
                ax.set_xlabel(date_col)
                
                # 优化日期标签显示
                if len(x_values) > 10:
                    step = max(1, len(x_values) // 10)
                    tick_indices = list(range(0, len(x_values), step))
                    tick_labels = [x_values.iloc[i] for i in tick_indices]
                    
                    ax.set_xticks(tick_indices)
                    ax.set_xticklabels(tick_labels, rotation=45)
                else:
                    ax.tick_params(axis='x', rotation=45)
            else:
                ax.plot(df_sql[first_numeric], marker='o', label=first_numeric, linewidth=2, color='#1f77b4')
                ax.set_xlabel('Index')
                
                # 优化X轴标签显示
                if len(df_sql) > 10:
                    step = max(1, len(df_sql) // 10)
                    tick_indices = list(range(0, len(df_sql), step))
                    ax.set_xticks(tick_indices)
                    ax.set_xticklabels([str(i) for i in tick_indices])
            
            ax.set_ylabel(first_numeric)
            ax.set_title('数据走势图')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    计算MACD指标
    :param data: 价格数据（通常是收盘价）
    :param fast: 快速EMA周期
    :param slow: 慢速EMA周期
    :param signal: 信号线周期
    :return: MACD线, 信号线, 柱状图
    """
    exp1 = data.ewm(span=fast).mean()
    exp2 = data.ewm(span=slow).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


@register_tool('macd_stock')
class MacdStock(BaseTool):
    description = '使用MACD指标分析股票交易信号，并计算过去一年的买卖点和收益率'
    parameters = [
        {
            'name': 'ts_code',
            'type': 'string',
            'description': '股票代码',
            'required': True
        }
    ]

    def call(self, params: str, **kwargs) -> List[Dict[str, Any]]:
        """
        使用MACD策略分析股票
        :param params: JSON字符串，包含ts_code参数
        :param kwargs: 其他参数
        :return: 交易信号和收益分析结果
        """
        import sqlite3
        from datetime import datetime, timedelta
        
        params_dict = json.loads(params)
        ts_code = params_dict.get('ts_code')
        
        if not ts_code:
            return [{'error': '股票代码(ts_code)是必填参数'}]
        
        try:
            # 连接数据库并获取股票数据
            connection = sqlite3.connect('stock_data.db')
            cursor = connection.cursor()
            
            # 计算一年前的日期
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=365)
            
            # 查询股票数据，获取过去一年的数据
            query = """
            SELECT trade_date, close 
            FROM stock_history 
            WHERE ts_code = ? AND trade_date >= ? 
            ORDER BY trade_date ASC
            """
            cursor.execute(query, (ts_code, start_date.strftime('%Y-%m-%d')))
            result = cursor.fetchall()
            
            if not result:
                return [{'error': f'找不到股票 {ts_code} 过去一年的数据，请检查股票代码是否正确'}]
            
            # 将结果转换为DataFrame
            df = pd.DataFrame(result, columns=['trade_date', 'close'])
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            # 检查数据量是否足够
            if len(df) < 50:
                return [{'error': f'股票 {ts_code} 的历史数据不足，无法进行MACD分析'}]
            
            # 计算MACD指标
            df['macd'], df['signal'], df['histogram'] = calculate_macd(df['close'])
            
            # 识别买卖点
            buy_signals = []
            sell_signals = []
            
            # MACD线上穿信号线为买入信号，下穿为卖出信号
            for i in range(1, len(df)):
                # 金叉：MACD线上穿信号线
                if df['macd'].iloc[i-1] <= df['signal'].iloc[i-1] and df['macd'].iloc[i] > df['signal'].iloc[i]:
                    buy_signals.append({
                        'date': df['trade_date'].iloc[i].strftime('%Y-%m-%d'),
                        'price': df['close'].iloc[i],
                        'type': 'BUY'
                    })
                # 死叉：MACD线下穿信号线
                elif df['macd'].iloc[i-1] >= df['signal'].iloc[i-1] and df['macd'].iloc[i] < df['signal'].iloc[i]:
                    sell_signals.append({
                        'date': df['trade_date'].iloc[i].strftime('%Y-%m-%d'),
                        'price': df['close'].iloc[i],
                        'type': 'SELL'
                    })
           
            
            # 如果最后一个信号是买入而没有对应的卖出，则添加到最后一天卖出
            if buy_signals and (not sell_signals or buy_signals[-1]['date'] > sell_signals[-1]['date']):
                sell_signals.append({
                    'date': df['trade_date'].iloc[-1].strftime('%Y-%m-%d'),
                    'price': df['close'].iloc[-1],
                    'type': 'SELL'
                })
            
            # 计算收益率（假设初始资金10000元，满仓交易）
            initial_amount = 10000
            current_amount = initial_amount
            transactions = []
            
            for i, buy_signal in enumerate(buy_signals):
                if i < len(sell_signals):  # 确保有对应的卖出信号
                    sell_signal = sell_signals[i]
                    
                    # 计算购买股数（满仓）
                    shares = current_amount / buy_signal['price']
                    
                    # 卖出后得到的资金
                    sold_amount = shares * sell_signal['price']
                    
                    # 计算收益率
                    profit_rate = (sold_amount - current_amount) / current_amount * 100
                    
                    transaction = {
                        'buy_date': buy_signal['date'],
                        'buy_price': buy_signal['price'],
                        'sell_date': sell_signal['date'],
                        'sell_price': sell_signal['price'],
                        'initial_amount': round(current_amount, 2),
                        'final_amount': round(sold_amount, 2),
                        'profit_rate': round(profit_rate, 2)
                    }
                    
                    transactions.append(transaction)
                    current_amount = sold_amount  # 更新当前金额用于下次交易
            
            # 计算总体收益率
            total_return_rate = (current_amount - initial_amount) / initial_amount * 100 if initial_amount != 0 else 0
            
            # 生成MACD图表
            save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
            os.makedirs(save_dir, exist_ok=True)
            filename = f'macd_analysis_{ts_code}_{int(time.time() * 1000)}.png'
            save_path = os.path.join(save_dir, filename)
            
            self._generate_macd_chart(df, ts_code, save_path, buy_signals, sell_signals)
            
            # 构造返回结果
            img_path = os.path.join('image_show', filename)
            img_md = f'![MACD股票分析图]({img_path})'
            
            result_summary = {
                'stock_code': ts_code,
                'total_transactions': len(transactions),
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'transactions': transactions,
                'initial_amount': initial_amount,
                'final_amount': round(current_amount, 2),
                'total_return_rate': round(total_return_rate, 2),
                'chart': img_md
            }
            
            return [result_summary]
        
        except Exception as e:
            return [{'error': f'执行MACD分析时出错: {str(e)}'}]
        finally:
            if 'connection' in locals():
                connection.close()
    
    def _generate_macd_chart(self, df, ts_code, save_path, buy_signals, sell_signals):
        """
        生成MACD分析图表
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # 上图：价格走势和买卖点
        ax1.plot(df['trade_date'], df['close'], label='收盘价', color='black', linewidth=1)
        
        # 标记买入点
        if buy_signals:
            buy_dates = [pd.to_datetime(bs['date']) for bs in buy_signals]
            buy_prices = [bs['price'] for bs in buy_signals]
            ax1.scatter(buy_dates, buy_prices, color='red', s=50, label='买入点', marker='^', zorder=5)
        
        # 标记卖出点
        if sell_signals:
            sell_dates = [pd.to_datetime(ss['date']) for ss in sell_signals]
            sell_prices = [ss['price'] for ss in sell_signals]
            ax1.scatter(sell_dates, sell_prices, color='green', s=50, label='卖出点', marker='v', zorder=5)
        
        ax1.set_title(f'{ts_code} 股票价格及MACD交易信号')
        ax1.set_ylabel('价格')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 下图：MACD线和信号线
        ax2.plot(df['trade_date'], df['macd'], label='MACD', color='blue', linewidth=1)
        ax2.plot(df['trade_date'], df['signal'], label='信号线', color='orange', linewidth=1)
        ax2.bar(df['trade_date'], df['histogram'], label='柱状图', alpha=0.3)
        
        ax2.set_xlabel('日期')
        ax2.set_ylabel('MACD值')
        ax2.set_title('MACD指标')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # 优化日期标签显示
        for ax in [ax1, ax2]:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


@register_tool('arima_stock')
class ArimaStock(BaseTool):
    description = '使用ARIMA模型对未来N天的股票价格进行预测'
    parameters = [
        {
            'name': 'ts_code',
            'type': 'string',
            'description': '股票代码',
            'required': True
        },
        {
            'name': 'n',
            'type': 'integer',
            'description': '预测天数',
            'required': True
        }
    ]

    def call(self, params: str, **kwargs) -> List[Dict[str, Any]]:
        """
        使用ARIMA模型预测股票价格
        :param params: JSON字符串，包含ts_code和n参数
        :param kwargs: 其他参数
        :return: 预测结果
        """
        import sqlite3
        import warnings
        from datetime import datetime, timedelta
        from statsmodels.tsa.arima.model import ARIMA
        
        warnings.filterwarnings("ignore")  # 忽略警告信息
        
        params_dict = json.loads(params)
        ts_code = params_dict.get('ts_code')
        n = params_dict.get('n')
        
        if not ts_code:
            return [{'error': '股票代码(ts_code)是必填参数'}]
        
        if not n or n <= 0:
            return [{'error': '预测天数(n)必须是正整数'}]
        
        try:
            # 连接数据库并获取股票数据
            connection = sqlite3.connect('stock_data.db')
            cursor = connection.cursor()
            
            # 计算一年前的日期
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=365)
            
            # 查询股票数据，获取过去一年的数据
            query = """
            SELECT trade_date, close 
            FROM stock_history 
            WHERE ts_code = ? AND trade_date >= ? 
            ORDER BY trade_date ASC
            """
            cursor.execute(query, (ts_code, start_date.strftime('%Y-%m-%d')))
            result = cursor.fetchall()
            
            if not result:
                return [{'error': f'找不到股票 {ts_code} 的数据或数据不足，请检查股票代码是否正确'}]
            
            # 将结果转换为DataFrame
            df = pd.DataFrame(result, columns=['trade_date', 'close'])
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            # 检查数据量是否足够
            if len(df) < 10:
                return [{'error': f'股票 {ts_code} 的历史数据不足，无法进行ARIMA预测'}]
            
            # 对时间序列数据进行预处理
            df = df.sort_values('trade_date')
            df.reset_index(drop=True, inplace=True)
            
            # 使用收盘价进行ARIMA建模
            close_prices = df['close'].values
            
            # 拟合ARIMA模型，使用(5,1,5)参数
            model = ARIMA(close_prices, order=(5, 1, 5))
            fitted_model = model.fit()
            
            # 进行预测
            forecast_result = fitted_model.forecast(steps=n)
            forecast_conf_int = fitted_model.get_forecast(steps=n).conf_int()
            
            # 生成未来日期
            future_dates = []
            last_date = df['trade_date'].values[-1]  # 使用.values获取实际的日期值
            for i in range(1, n + 1):
                future_date = pd.to_datetime(last_date) + timedelta(days=i)
                future_dates.append(future_date.strftime('%Y-%m-%d'))
            
            # 准备返回结果
            prediction_results = []
            for i in range(n):
                prediction_data = {
                    'date': future_dates[i],
                    'predicted_price': round(float(forecast_result[i]), 2),
                    'lower_bound': round(float(forecast_conf_int[i, 0]), 2),
                    'upper_bound': round(float(forecast_conf_int[i, 1]), 2)
                }
                prediction_results.append(prediction_data)
            
            # 创建包含历史数据和预测数据的完整DataFrame用于绘图
            historical_data = df[['trade_date', 'close']].rename(columns={'close': 'actual_price'})
            historical_data['predicted_price'] = np.nan
            historical_data['type'] = 'historical'
            
            # 创建预测数据的DataFrame
            predicted_df = pd.DataFrame({
                'trade_date': pd.to_datetime(future_dates),
                'actual_price': np.nan,
                'predicted_price': forecast_result,
                'type': 'predicted'
            })
            
            # 合并数据
            combined_df = pd.concat([historical_data, predicted_df], ignore_index=True)
            
            # 生成预测结果图表
            save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
            os.makedirs(save_dir, exist_ok=True)
            filename = f'arima_prediction_{ts_code}_{int(time.time() * 1000)}.png'
            save_path = os.path.join(save_dir, filename)
            
            self._generate_arima_chart(combined_df, ts_code, save_path)
            
            # 构造返回结果
            img_path = os.path.join('image_show', filename)
            img_md = f'![ARIMA股票预测图]({img_path})'
            
            result_summary = {
                'stock_code': ts_code,
                'prediction_days': n,
                'predictions': prediction_results,
                'chart': img_md
            }
            
            return [result_summary]
        
        except Exception as e:
            return [{'error': f'执行ARIMA预测时出错: {str(e)}'}]
        finally:
            if 'connection' in locals():
                connection.close()
    
    def _generate_arima_chart(self, df, ts_code, save_path):
        """
        生成ARIMA预测图表
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制历史数据
        historical_data = df[df['type'] == 'historical']
        if not historical_data.empty:
            ax.plot(historical_data['trade_date'], historical_data['actual_price'], 
                   label='历史价格', color='#1f77b4', linewidth=2)
        
        # 绘制预测数据
        predicted_data = df[df['type'] == 'predicted']
        if not predicted_data.empty:
            ax.plot(predicted_data['trade_date'], predicted_data['predicted_price'], 
                   label='预测价格', color='#ff7f0e', linestyle='--', linewidth=2, marker='o')
        
        ax.set_xlabel('日期')
        ax.set_ylabel('价格')
        ax.set_title(f'{ts_code} 股票价格ARIMA预测结果')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 优化日期标签显示
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()





# 引入布林带检测功能
from boll_detection import BollingerBandDetector

@register_tool('boll_stock_main')
class BollStockMain(BaseTool):
    description = '使用布林带检测股票超买超卖点，并计算基于信号的交易收益率'
    parameters = [
        {
            'name': 'ts_code',
            'type': 'string',
            'description': '股票代码',
            'required': True
        },
        {
            'name': 'start_date',
            'type': 'string',
            'description': '开始日期，格式YYYY-MM-DD，默认为一年前',
            'required': False
        },
        {
            'name': 'end_date',
            'type': 'string',
            'description': '结束日期，格式YYYY-MM-DD，默认为今天',
            'required': False
        },
        {
            'name': 'window',
            'type': 'integer',
            'description': '移动平均窗口大小，默认20',
            'required': False
        },
        {
            'name': 'num_std',
            'type': 'number',
            'description': '标准差倍数，默认2',
            'required': False
        },
        {
            'name': 'initial_amount',
            'type': 'number',
            'description': '初始资金，默认10000元',
            'required': False
        }
    ]

    def call(self, params: str, **kwargs) -> List[Dict[str, Any]]:
        """
        使用布林带策略分析股票
        :param params: JSON字符串，包含参数
        :param kwargs: 其他参数
        :return: 布林带分析结果
        """
        detector = BollingerBandDetector()
        return detector.call(params, **kwargs)


# 引入Prophet分析功能
try:
    from prophet_analysis import ProphetAnalysisTool
except ImportError as e:
    print(f"Warning: Could not import Prophet analysis tool: {e}")
    ProphetAnalysisTool = None


def main():
    # 使用 DeepSeek API（请替换为你的 API Key）
    llm_cfg = {
        'model': 'deepseek-v4-pro',
        'model_server': 'https://api.deepseek.com/v1',
        'api_key': os.getenv('DEEPSEEK_API_KEY', 'sk-88636fbb132646f4b981e71ac5f6b2a1'),
    }

    system_instruction = '''
    您是一个专业的股票查询和分析助手，能够通过执行SQL查询来获取股票历史数据、分析趋势并比较股票表现，并生成可视化图表。

    您可以执行以下操作：
    1. 查询股票历史数据 - 使用ExcSql工具执行SELECT查询
    2. 分析股票趋势 - 使用ExcSql工具执行统计分析查询
    3. 比较股票表现 - 使用ExcSql工具执行多股票比较查询
    4. 预测股票价格 - 使用arima_stock工具对未来N天的股票价格进行预测
    5. MACD技术分析 - 使用macd_stock工具分析股票的MACD交易信号及历史收益率
    6. 布林带分析 - 使用boll_stock_main工具检测股票的超买超卖点及基于信号的历史收益率
    7. Prophet周期性分析 - 使用prophet_analysis工具对股票价格进行周期性分析，包括趋势、周和年季节性分析
    
    ExcSql工具会自动根据查询结果生成相应的图表。
    arima_stock工具使用ARIMA模型对未来股票价格进行预测。
    macd_stock工具使用MACD指标分析股票的买卖点和历史收益率，需要提供股票代码(ts_code)
    boll_stock_main工具使用布林带检测股票的超买超卖点和基于信号的收益率，需要提供股票代码(ts_code)，可选参数包括开始日期(start_date)、结束日期(end_date)、移动平均窗口(window)、标准差倍数(num_std)和初始资金(initial_amount)
    prophet_analysis工具使用Prophet模型对股票价格进行周期性分析，识别趋势、每周和每年的周期性模式，需要提供股票代码(ts_code)，可选参数包括开始日期(start_date)和结束日期(end_date)

    以下是股票历史数据表的结构（使用SQLite数据库）：
    CREATE TABLE stock_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_code TEXT NOT NULL,          -- 股票代码
        trade_date TEXT NOT NULL,       -- 交易日期
        open REAL NOT NULL,             -- 开盘价
        high REAL NOT NULL,             -- 最高价
        low REAL NOT NULL,              -- 最低价
        close REAL NOT NULL,            -- 收盘价
        pre_close REAL,                 -- 前收盘价
        change REAL,                    -- 涨跌额
        pct_chg REAL,                   -- 涨跌幅(%)
        vol INTEGER,                    -- 成交量(手)
        amount REAL,                    -- 成交额(千元)
        stock_name TEXT NOT NULL        -- 股票名称
    );
    
    -- 创建索引
    CREATE INDEX idx_trade_date ON stock_history (trade_date);      -- 交易日期索引
    CREATE INDEX idx_ts_code ON stock_history (ts_code);           -- 股票代码索引
    CREATE INDEX idx_stock_name ON stock_history (stock_name);     -- 股票名称索引
    
    数据库包含以下股票：
    - 600519.SH: 贵州茅台
    - 000858.SZ: 五粮液
    - 601211.SH: 国泰君安
    - 688981.SH: 中芯国际
    
    注意：
    - 数据库使用SQLite，SQL语法遵循SQLite规范
    - 数值类型使用REAL、INTEGER，而非DECIMAL
    - 字符串类型使用TEXT，而非VARCHAR
    - 时间日期存储为TEXT格式，格式为YYYY-MM-DD（如'2025-12-19'代表2025年12月19日）
    - 执行日期比较时，可以直接使用日期字符串，例如：trade_date >= '2025-01-01' 或 trade_date BETWEEN '2025-01-01' AND '2025-12-19'
    - 当使用DATE函数进行相对日期计算时（如DATE('now', '-1 month')），系统会自动将其转换为正确的YYYY-MM-DD格式
    - 支持的相对日期函数包括但不限于：DATE('now'), DATE('now', '-1 day'), DATE('now', '+1 day'), DATE('now', '-1 week'), DATE('now', '-2 weeks'), DATE('now', '-1 month'), DATE('now', '-3 months'), DATE('now', '-6 months'), DATE('now', '-1 year')
    - 使用双引号引用标识符，而不是反引号
    - arima_stock工具使用ARIMA(5,1,5)模型对未来N天的股票价格进行预测，需要提供股票代码(ts_code)和预测天数(n)
    - prophet_analysis工具需要事先安装Prophet库（pip install prophet），它使用Prophet模型进行周期性分析，能够识别股票价格的趋势、每周和每年的季节性模式
    
    在回答时，请：
    - 提供清晰、准确的数据和分析结果
    - 使用易懂的语言解释复杂的金融概念
    - 在适当的时候提醒投资风险
    - 当数据不足以回答问题时，诚实地告知用户
    - 每当ExcSql工具或arima_stock工具或prophet_analysis工具返回markdown表格和图片时，您必须原样输出工具返回的全部内容（包括图片markdown），不要只总结表格，也不要省略图片。这样用户才能直接看到表格和图片。
    '''

    tools = [
        'ExcSql',
        'arima_stock',
        'macd_stock',
        'boll_stock_main',
        'prophet_analysis',
    ]

    bot = Assistant(
        llm=llm_cfg,
        name='股票查询和分析助手',
        description='股票数据查询与分析',
        system_message=system_instruction,  # 使用 system_message 替代 system_instruction
        function_list=tools,
        files = ['./faq.txt']
    )
    
    # 启动GUI界面
    from qwen_agent.gui import WebUI
    
    # 配置聊天界面建议
    chatbot_config = {
        'prompt.suggestions': [
            '查询贵州茅台最近一个月的股价走势',
            '查询2024年全年贵州茅台的收盘价走势',
            '对比2024年中芯国际和贵州茅台的涨跌幅',
            '使用ARIMA模型预测贵州茅台未来7天的价格',
            '预测600519.SH股票未来5天的收盘价',
            '使用MACD分析贵州茅台过去一年的买卖点',
            '分析000858.SZ股票的MACD交易信号和收益率',
            '使用布林带检测600519.SH股票的超买超卖点',
            '分析000858.SZ股票的布林带信号和收益率',
            '使用Prophet分析600519.SH股票的趋势和周期性',
            '分析000858.SZ股票的年度和周度季节性模式'
        ]
    }
    
    # 启动 WebUI
    WebUI(
        bot,
        chatbot_config=chatbot_config
    ).run()


if __name__ == '__main__':
    main()
