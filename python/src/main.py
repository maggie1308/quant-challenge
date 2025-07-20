# Точка входа для реализации
import os
import json
import time
import datetime
import argparse
import asyncio

def main():
    parser = argparse.ArgumentParser(description="Quant challenge runner")
    parser.add_argument("--mode", type=str, default="collect", choices=["collect", "analyze", "backtest", "report"], help="Run mode")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol (Bybit)")
    parser.add_argument("--minutes", type=int, default=10, help="How many minutes to collect data")
    parser.add_argument("--interval", type=float, default=1.0, help="Interval (seconds) between snapshots")
    args = parser.parse_args()

    if args.mode == "collect":
        asyncio.run(collect_orderbooks(symbol=args.symbol, minutes=args.minutes, interval=args.interval))
    elif args.mode == "analyze":
        analyze_orderbooks()
    elif args.mode == "backtest":
        backtest_strategy()
    elif args.mode == "report":
        generate_report()
    else:
        print("Unknown mode")

async def collect_orderbooks(symbol, minutes, interval):
    from infrastructure.adapters.bybit import BybitClient

    os.makedirs("data", exist_ok=True)
    filename = f"data/orderbook_{symbol}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    print(f"Сохраняем сырые данные в {filename}")

    client = BybitClient()
    snapshots = []
    n_steps = int(minutes * 60 / interval)
    try:
        for i in range(n_steps):
            t0 = time.time()
            try:
                snap = await client.fetch_orderbook_snapshot(symbol)
            except Exception as e:
                print(f"Ошибка при получении данных: {e}")
                snap = {}
            snapshots.append({
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "data": snap,
            })
            dt = time.time() - t0
            if dt < interval:
                await asyncio.sleep(interval - dt)
            print(f"Step {i+1}/{n_steps} done", end="\r")
    finally:
        await client.close()

    # Сохраняем всё в файл
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(snapshots, f, ensure_ascii=False, indent=2)
    print(f"\nСохранено {len(snapshots)} снапшотов.")

def analyze_orderbooks():
    import glob
    import pandas as pd
    from infrastructure.adapters.bybit import BybitClient
    import asyncio

    # Функция для вычисления midprice
    def get_midprice(bids, asks):
        if bids and asks:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            return (best_bid + best_ask) / 2
        return None

    # Найдём последний файл данных
    files = sorted(glob.glob("data/orderbook_*.json"))
    if not files:
        print("Нет файлов с данными в папке data/")
        return
    filename = files[-1]
    print(f"Анализируем файл: {filename}")

    with open(filename, "r", encoding="utf-8") as f:
        snapshots = json.load(f)

    # Извлекаем только bids/asks из каждого snapshot и сразу считаем midprice
    orderbooks = []
    prices = []
    for snap in snapshots:
        ts = snap["timestamp"]
        data = snap["data"]
        bids = data.get("b", [])
        asks = data.get("a", [])
        orderbooks.append({"timestamp": ts, "bids": bids, "asks": asks})
        prices.append(get_midprice(bids, asks))

    async def calc_deltas(orderbooks):
        client = BybitClient()
        deltas = []
        for i in range(1, len(orderbooks)):
            prev = orderbooks[i-1]
            curr = orderbooks[i]
            # эмулируем интерфейс BybitClient.calculate_delta
            delta = await client.calculate_delta(
                {"b": prev["bids"], "a": prev["asks"]},
                {"b": curr["bids"], "a": curr["asks"]}
            )
            deltas.append({
                "timestamp": curr["timestamp"],
                "delta": delta,
            })
        await client.close()
        return deltas

    deltas = asyncio.run(calc_deltas(orderbooks))

    # Сохраним в CSV, добавляя midprice к каждой дельте
    df = pd.DataFrame(deltas)
    df["midprice"] = prices[1:]  # midprice для каждого curr в дельтах (начиная со второго снапшота)
    csv_out = filename.replace(".json", "_deltas.csv")
    df.to_csv(csv_out, index=False)
    print(f"Сохранили дельты в {csv_out}")
    print(df.head())


def backtest_strategy():
    import matplotlib
    matplotlib.use("Agg")
    import glob
    import pandas as pd
    import matplotlib.pyplot as plt

    # Найдём последний файл дельт
    files = sorted(glob.glob("data/orderbook_*_deltas.csv"))
    if not files:
        print("Нет файлов с дельтами в папке data/")
        return
    filename = files[-1]
    print(f"Бэктестируем по: {filename}")

    df = pd.read_csv(filename)

    if "midprice" in df.columns and not df["midprice"].isnull().all():
        df["price"] = df["midprice"]
    threshold = 0.001

    def get_signal(delta, thr):
        if delta > thr:
            return 1
        elif delta < -thr:
            return -1
        else:
            return 0

    df["signal"] = df["delta"].apply(lambda x: get_signal(x, threshold))

    # Бэктест: считаем equity
    position = 0
    entry_price = None
    equity = [0]
    trades = []
    for i, row in df.iterrows():
        signal = row["signal"]
        price = row["price"]
        if signal != 0 and signal != position:
            if position != 0 and entry_price is not None:
                pnl = (price - entry_price) * position
                equity.append(equity[-1] + pnl)
                trades.append({"entry": entry_price, "exit": price, "side": position, "pnl": pnl})
            else:
                equity.append(equity[-1])
            entry_price = price
            position = signal
        else:
            equity.append(equity[-1])
    df["equity"] = equity[1:]

    # Сохраним результат
    result_csv = filename.replace("_deltas.csv", "_bt.csv")
    df.to_csv(result_csv, index=False)
    print(f"Бэктест сохранён в {result_csv}")

    # Визуализация
    plt.figure(figsize=(10, 4))
    plt.plot(df["equity"])
    plt.title("Equity Curve (Simple Strategy)")
    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    result_png = filename.replace("_deltas.csv", "_bt_equity.png")
    plt.savefig(result_png)
    print(f"График equity сохранён в {result_png}")
    # plt.show()

def generate_report():
    import glob
    import pandas as pd

    files = sorted(glob.glob("data/orderbook_*_bt.csv"))
    if not files:
        print("Нет bt-файлов для отчёта")
        return
    filename = files[-1]
    print(f"Генерируем отчёт по: {filename}")

    df = pd.read_csv(filename)
    num_signals = (df["signal"] != 0).sum()
    final_pnl = df["equity"].iloc[-1]
    max_drawdown = (df["equity"].cummax() - df["equity"]).max()

    print("=" * 32)
    print("ОТЧЁТ ПО СТРАТЕГИИ:")
    print(f"Число сигналов: {num_signals}")
    print(f"Финальный PnL: {final_pnl}")
    print(f"Максимальная просадка: {max_drawdown}")
    print("=" * 32)

if __name__ == "__main__":
    main()
