import json
import os
import decimal
from typing import List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime
import time


def create_prophet_analysis_tool():
    """
    创建Prophet股票周期性分析工具
    """
    try:
        from qwen_agent.tools.base import BaseTool, register_tool
        import matplotlib.pyplot as plt
        
        # 解决中文显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    except ImportError as e:
        print(f"无法导入必要模块: {e}")
        return None

    @register_tool('prophet_analysis')
    class ProphetAnalysis(BaseTool):
        description = '使用Prophet模型对股票价格进行周期性分析，包括趋势(Trend)、每周(Weekly)和每年(Yearly)的周期性分析'
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
            }
        ]

        def call(self, params: str, **kwargs) -> List[Dict[str, Any]]:
            """
            执行Prophet周期性分析
            :param params: JSON字符串，包含ts_code和其他可选参数
            :param kwargs: 其他参数
            :return: 分析结果
            """
            params_dict = json.loads(params)
            ts_code = params_dict.get('ts_code')
            
            if not ts_code:
                return [{'error': '股票代码(ts_code)是必填参数'}]

            try:
                # 检查是否安装了Prophet库
                try:
                    from prophet import Prophet
                except ImportError:
                    return [{'error': 'Prophet库未安装。请使用命令 "pip install prophet" 安装Prophet库后再使用此功能'}]

                # 计算默认日期范围
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
                
                # 检查是否传入了自定义日期
                if params_dict.get('start_date'):
                    start_date = datetime.strptime(params_dict.get('start_date'), '%Y-%m-%d')
                if params_dict.get('end_date'):
                    end_date = datetime.strptime(params_dict.get('end_date'), '%Y-%m-%d')

                # 连接数据库并获取股票数据
                connection = sqlite3.connect('stock_data.db')
                cursor = connection.cursor()

                # 查询股票数据
                query = """
                SELECT trade_date, close 
                FROM stock_history 
                WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ?
                ORDER BY trade_date ASC
                """
                cursor.execute(query, (ts_code, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
                result = cursor.fetchall()

                if not result:
                    return [{'error': f'找不到股票 {ts_code} 在指定日期范围内的数据，请检查股票代码和日期范围是否正确'}]

                # 将结果转换为DataFrame
                df = pd.DataFrame(result, columns=['trade_date', 'close'])
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                
                # 检查数据量是否足够
                if len(df) < 30:
                    return [{'error': f'股票 {ts_code} 在指定日期范围内的历史数据不足，无法进行Prophet周期性分析'}]

                # 按日期排序并重置索引
                df = df.sort_values('trade_date').reset_index(drop=True)

                # 检查数据是否有缺失日期
                if len(df) > 1:
                    date_range = pd.date_range(start=df['trade_date'].min(), end=df['trade_date'].max(), freq='D')
                    missing_dates = set(date_range) - set(df['trade_date'])
                    
                    # 如果缺失日期较多，可能会影响Prophet的准确性
                    if len(missing_dates) > len(date_range) * 0.3:  # 如果缺失超过30%的数据
                        print(f"警告：股票 {ts_code} 在指定日期范围内有大量缺失数据")

                # 为了Prophet的要求，重命名列
                prophet_df = df[['trade_date', 'close']].rename(columns={'trade_date': 'ds', 'close': 'y'})

                # 创建Prophet模型并拟合
                model = Prophet(
                    daily_seasonality=False,  # 不启用日季节性
                    weekly_seasonality=True,  # 启用周季节性
                    yearly_seasonality=True,  # 启用年季节性
                    interval_width=0.95     # 置信区间宽度
                )
                
                # 添加月度季节性
                model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                
                # 拟合模型
                model.fit(prophet_df)

                # 创建未来数据框用于分解
                future = model.make_future_dataframe(periods=0)  # 不预测未来，只分析历史数据
                
                # 进行预测（实际是拟合）
                forecast = model.predict(future)
                
                # 生成组件图
                save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
                os.makedirs(save_dir, exist_ok=True)
                filename = f'prophet_components_{ts_code}_{int(time.time() * 1000)}.png'
                save_path = os.path.join(save_dir, filename)
                
                # 生成Prophet组件图
                try:
                    fig = model.plot_components(forecast)
                    fig.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)  # 关闭图形以释放内存
                except Exception as e:
                    return [{'error': f'生成Prophet组件图时出错: {str(e)}'}]
                
                # 生成整体趋势图
                trend_filename = f'prophet_trend_{ts_code}_{int(time.time() * 1000)}.png'
                trend_save_path = os.path.join(save_dir, trend_filename)
                
                try:
                    fig2 = model.plot(forecast)
                    fig2.savefig(trend_save_path, dpi=300, bbox_inches='tight')
                    plt.close(fig2)
                except Exception as e:
                    return [{'error': f'生成Prophet趋势图时出错: {str(e)}'}]

                # 提取趋势、周季节性和年季节性的统计信息
                trend_change = self._analyze_trend_change(forecast)
                weekly_effect = self._analyze_weekly_effect(forecast, df)
                yearly_effect = self._analyze_yearly_effect(df)

                # 构造返回结果
                img_path = os.path.join('image_show', filename)
                trend_img_path = os.path.join('image_show', trend_filename)
                img_md = f'![Prophet组件分析图]({img_path})'
                trend_img_md = f'![Prophet趋势图]({trend_img_path})'

                result_summary = {
                    'stock_code': ts_code,
                    'analysis_period': f"{start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}",
                    'data_points_count': len(df),
                    'trend_analysis': trend_change,
                    'weekly_analysis': weekly_effect,
                    'yearly_analysis': yearly_effect,
                    'components_chart': img_md,
                    'trend_chart': trend_img_md
                }

                return [result_summary]

            except Exception as e:
                return [{'error': f'执行Prophet分析时出错: {str(e)}'}]
            finally:
                if 'connection' in locals():
                    connection.close()
        
        def _analyze_trend_change(self, forecast_df):
            """
            分析趋势变化
            """
            try:
                # 获取趋势值
                trend_values = forecast_df['trend'].dropna()
                
                if len(trend_values) < 2:
                    return {'message': '数据不足，无法分析趋势变化'}
                
                start_trend = trend_values.iloc[0]
                end_trend = trend_values.iloc[-1]
                
                trend_change = ((end_trend - start_trend) / start_trend) * 100 if start_trend != 0 else 0
                
                trend_direction = "上升" if trend_change > 0 else "下降" if trend_change < 0 else "平稳"
                
                return {
                    'direction': trend_direction,
                    'change_percentage': round(trend_change, 2),
                    'start_value': round(start_trend, 2),
                    'end_value': round(end_trend, 2)
                }
            except Exception as e:
                return {'message': f'趋势分析出错: {str(e)}'}

        def _analyze_weekly_effect(self, forecast_df, original_df):
            """
            分析周季节性效应
            """
            try:
                # Prophet模型拟合后，我们可以获取季节性分量
                # 从forecast_df中提取周季节性分量
                if 'weekly' not in forecast_df.columns:
                    return {'message': '周季节性分析失败，Prophet模型未包含weekly分量'}
                
                weekly_effects = forecast_df['weekly'].dropna()
                
                if len(weekly_effects) == 0:
                    return {'message': '周季节性分析失败，无有效数据'}
                
                avg_positive_weekly = weekly_effects[weekly_effects > 0].mean() if any(weekly_effects > 0) else 0
                avg_negative_weekly = weekly_effects[weekly_effects < 0].mean() if any(weekly_effects < 0) else 0
                max_weekly_effect = weekly_effects.max()
                min_weekly_effect = weekly_effects.min()
                
                return {
                    'avg_positive_effect': round(avg_positive_weekly, 2) if not pd.isna(avg_positive_weekly) else 0,
                    'avg_negative_effect': round(avg_negative_weekly, 2) if not pd.isna(avg_negative_weekly) else 0,
                    'max_effect': round(max_weekly_effect, 2),
                    'min_effect': round(min_weekly_effect, 2)
                }
            except Exception as e:
                return {'message': f'周季节性分析出错: {str(e)}'}

        def _analyze_yearly_effect(self, original_df):
            """
            分析年季节性效应
            """
            try:
                # 分析原始数据中的年度模式
                original_df['month'] = original_df['trade_date'].dt.month
                monthly_avg = original_df.groupby('month')['close'].mean()
                
                highest_month = monthly_avg.idxmax()
                lowest_month = monthly_avg.idxmin()
                highest_price = monthly_avg.max()
                lowest_price = monthly_avg.min()
                
                return {
                    'highest_month': int(highest_month),
                    'highest_month_avg_price': round(highest_price, 2),
                    'lowest_month': int(lowest_month),
                    'lowest_month_avg_price': round(lowest_price, 2)
                }
            except Exception as e:
                return {'message': f'年季节性分析出错: {str(e)}'}

    return ProphetAnalysis


# 创建工具类
ProphetAnalysisTool = create_prophet_analysis_tool()
