import json
import os
import decimal
from typing import List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import time
import sqlite3

from qwen_agent.tools.base import BaseTool, register_tool

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def calculate_bollinger_bands(data, window=20, num_std=2):
    """
    计算布林带
    :param data: 价格数据（通常是收盘价）
    :param window: 移动平均窗口（默认20日）
    :param num_std: 标准差倍数（默认2倍）
    :return: 中轨(MA)、上轨(UB)、下轨(LB)
    """
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    return rolling_mean, upper_band, lower_band


def detect_bollinger_bands_signals(df, window=20, num_std=2):
    """
    检测布林带超买超卖信号
    :param df: 包含'date'和'close'列的DataFrame
    :param window: 布林带移动平均窗口
    :param num_std: 标准差倍数
    :return: 包含信号的DataFrame
    """
    # 计算布林带
    df['ma'], df['upper_band'], df['lower_band'] = calculate_bollinger_bands(df['close'], window, num_std)
    
    # 检测信号
    # 股价突破上轨为超买信号
    df['oversold'] = df['close'] <= df['lower_band']
    # 股价跌破下轨为超卖信号
    df['overbought'] = df['close'] >= df['upper_band']
    
    # 获取具体的超买超卖点
    oversold_points = df[df['oversold']].copy()
    overbought_points = df[df['overbought']].copy()
    
    return oversold_points, overbought_points, df


def calculate_returns_from_signals(initial_amount, df, oversold_points, overbought_points):
    """
    根据超买超卖信号计算收益率
    :param initial_amount: 初始金额
    :param df: 完整数据
    :param oversold_points: 超卖点
    :param overbought_points: 超买点
    :return: 交易记录和最终收益率
    """
    current_amount = initial_amount
    transactions = []
    
    # 确保信号点按照时间顺序排列
    oversold_list = oversold_points.sort_values('trade_date').to_dict('records')
    overbought_list = overbought_points.sort_values('trade_date').to_dict('records')
    
    # 交替处理超卖和超买信号（先超卖买入，再超买卖出）
    buy_idx = 0
    sell_idx = 0
    
    while buy_idx < len(oversold_list) and sell_idx < len(overbought_list):
        buy_point = oversold_list[buy_idx]
        sell_point = overbought_list[sell_idx]
        
        # 确保买入日期在卖出日期之前
        if buy_point['trade_date'] < sell_point['trade_date']:
            # 执行买入
            shares = current_amount / buy_point['close']
            
            # 执行卖出
            sold_amount = shares * sell_point['close']
            
            # 计算收益率
            profit_rate = (sold_amount - current_amount) / current_amount * 100
            
            transaction = {
                'buy_date': str(buy_point['trade_date']),
                'buy_price': buy_point['close'],
                'sell_date': str(sell_point['trade_date']),
                'sell_price': sell_point['close'],
                'initial_amount': round(current_amount, 2),
                'final_amount': round(sold_amount, 2),
                'profit_rate': round(profit_rate, 2)
            }
            
            transactions.append(transaction)
            current_amount = sold_amount  # 更新当前金额用于下次交易
            
            # 移动到下一个买入和卖出点
            buy_idx += 1
            sell_idx += 1
        elif buy_point['trade_date'] > sell_point['trade_date']:
            # 如果卖出日期早于任何买入日期，跳过这个卖出信号
            sell_idx += 1
        else:
            # 如果日期相同，移动到下一个
            buy_idx += 1
            sell_idx += 1
    
    # 计算总体收益率
    total_return_rate = (current_amount - initial_amount) / initial_amount * 100 if initial_amount != 0 else 0
    
    return transactions, current_amount, total_return_rate


def generate_bollinger_bands_chart(df, oversold_points, overbought_points, ts_code, save_path):
    """
    生成布林带分析图表
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制价格线
    ax.plot(df['trade_date'], df['close'], label='收盘价', color='black', linewidth=1)
    
    # 绘制布林带
    ax.plot(df['trade_date'], df['ma'], label='中轨(MA20)', color='blue', linestyle='--')
    ax.plot(df['trade_date'], df['upper_band'], label='上轨', color='red', linestyle='--')
    ax.plot(df['trade_date'], df['lower_band'], label='下轨', color='green', linestyle='--')
    
    # 标记超卖点（买入信号）
    if not oversold_points.empty:
        ax.scatter(oversold_points['trade_date'], oversold_points['close'], 
                  color='green', s=50, label='超卖点(买入信号)', marker='^', zorder=5)
    
    # 标记超买点（卖出信号）
    if not overbought_points.empty:
        ax.scatter(overbought_points['trade_date'], overbought_points['close'], 
                  color='red', s=50, label='超买点(卖出信号)', marker='v', zorder=5)
    
    ax.set_title(f'{ts_code} 股票价格及布林带分析')
    ax.set_xlabel('日期')
    ax.set_ylabel('价格')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 优化日期标签显示
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def bollinger_bands_analysis(ts_code, start_date=None, end_date=None, window=20, num_std=2, initial_amount=10000):
    """
    布林带分析主函数
    :param ts_code: 股票代码
    :param start_date: 开始日期（格式：YYYY-MM-DD），如果不提供则默认为一年前
    :param end_date: 结束日期（格式：YYYY-MM-DD），如果不提供则默认为今天
    :param window: 布林带移动平均窗口
    :param num_std: 标准差倍数
    :param initial_amount: 初始资金，默认10000元
    :return: 分析结果
    """
    try:
        # 连接数据库
        connection = sqlite3.connect('stock_data.db')
        cursor = connection.cursor()
        
        # 如果没有提供日期范围，则默认使用过去一年
        if not start_date:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=365)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 查询股票数据
        query = """
        SELECT trade_date, close 
        FROM stock_history 
        WHERE ts_code = ? AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date ASC
        """
        cursor.execute(query, (ts_code, start_date, end_date))
        result = cursor.fetchall()
        
        if not result:
            return {'error': f'找不到股票 {ts_code} 在指定日期范围内的数据'}
        
        # 转换为DataFrame
        df = pd.DataFrame(result, columns=['trade_date', 'close'])
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        if len(df) < window:
            return {'error': f'股票 {ts_code} 在指定日期范围内的数据不足，至少需要{window}个交易日的数据'}
        
        # 检测布林带信号
        oversold_points, overbought_points, full_df = detect_bollinger_bands_signals(df, window, num_std)
        
        # 计算收益率
        transactions, final_amount, total_return_rate = calculate_returns_from_signals(
            initial_amount, df, oversold_points, overbought_points)
        
        # 生成图表
        save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
        os.makedirs(save_dir, exist_ok=True)
        filename = f'bollinger_bands_{ts_code}_{int(time.time() * 1000)}.png'
        save_path = os.path.join(save_dir, filename)
        
        generate_bollinger_bands_chart(full_df, oversold_points, overbought_points, ts_code, save_path)
        
        # 返回结果
        img_path = os.path.join('image_show', filename)
        img_md = f'![布林带股票分析图]({img_path})'
        
        result_summary = {
            'stock_code': ts_code,
            'period': f'{start_date} 至 {end_date}',
            'window': window,
            'num_std': num_std,
            'total_oversold_signals': len(oversold_points),
            'total_overbought_signals': len(overbought_points),
            'successful_transactions': len(transactions),
            'transactions': transactions,
            'initial_amount': initial_amount,
            'final_amount': round(final_amount, 2),
            'total_return_rate': round(total_return_rate, 2),
            'chart': img_md,
            'oversold_dates': [str(date) for date in oversold_points['trade_date']] if not oversold_points.empty else [],
            'overbought_dates': [str(date) for date in overbought_points['trade_date']] if not overbought_points.empty else []
        }
        
        return result_summary
        
    except Exception as e:
        return {'error': f'执行布林带分析时出错: {str(e)}'}
    finally:
        if 'connection' in locals():
            connection.close()


@register_tool('boll_stock')
class BollingerBandDetector(BaseTool):
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
            'description': '开始日期，格式YYYY-MM-DD，如不提供则默认为一年前',
            'required': False
        },
        {
            'name': 'end_date',
            'type': 'string',
            'description': '结束日期，格式YYYY-MM-DD，如不提供则默认为今天',
            'required': False
        },
        {
            'name': 'window',
            'type': 'integer',
            'description': '布林带移动平均窗口，默认20',
            'required': False
        },
        {
            'name': 'num_std',
            'type': 'number',
            'description': '布林带标准差倍数，默认2.0',
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
        使用布林带策略检测超买超卖点
        :param params: JSON字符串，包含ts_code等参数
        :param kwargs: 其他参数
        :return: 交易信号和收益分析结果
        """
        params_dict = json.loads(params)
        ts_code = params_dict.get('ts_code')
        start_date = params_dict.get('start_date')
        end_date = params_dict.get('end_date')
        window = params_dict.get('window', 20)
        num_std = params_dict.get('num_std', 2)
        initial_amount = params_dict.get('initial_amount', 10000)
        
        if not ts_code:
            return [{'error': '股票代码(ts_code)是必填参数'}]
        
        # 调用布林带分析函数
        result = bollinger_bands_analysis(ts_code, start_date, end_date, window, num_std, initial_amount)
        
        if 'error' in result:
            return [{'error': result['error']}]
        else:
            return [result]
