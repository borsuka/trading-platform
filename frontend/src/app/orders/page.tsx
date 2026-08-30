"use client";

import { Shell } from "@/components/Shell";
import {
  Disclaimer,
  Empty,
  ErrorNotice,
  Money,
  Notice,
  PageHeader,
  StatusBadge,
  useAsync,
} from "@/components/ui";
import { get } from "@/lib/api";

type Order = {
  id: string;
  client_order_id: string;
  exchange_order_id: string | null;
  symbol: string;
  side: string;
  order_type: string;
  status: string;
  quantity: number;
  price: number | null;
  filled_quantity: number;
  average_fill_price: number;
  fees_paid: number;
  reject_reason: string | null;
  created_at: string;
};

export default function OrdersPage() {
  return (
    <Shell>
      <Orders />
    </Shell>
  );
}

function Orders() {
  const orders = useAsync(() => get<Order[]>("/api/v1/orders?limit=200"));

  return (
    <>
      <PageHeader
        title="Orders"
        description="Every order the platform has submitted, including rejections."
        actions={
          <button type="button" onClick={orders.reload}>
            Refresh
          </button>
        }
      />

      <ErrorNotice error={orders.error} />

      {orders.data && orders.data.length > 0 ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Created</th>
                <th>Symbol</th>
                <th>Side</th>
                <th>Type</th>
                <th className="num">Quantity</th>
                <th className="num">Filled</th>
                <th className="num">Avg price</th>
                <th className="num">Fees</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {orders.data.map((order) => (
                <tr key={order.id}>
                  <td className="mono muted">{new Date(order.created_at).toLocaleString()}</td>
                  <td className="mono">{order.symbol}</td>
                  <td>{order.side}</td>
                  <td className="muted">{order.order_type.replace(/_/g, " ")}</td>
                  <td className="num">{order.quantity}</td>
                  <td className="num">{order.filled_quantity}</td>
                  <td className="num">
                    {order.average_fill_price ? order.average_fill_price.toFixed(2) : "—"}
                  </td>
                  <td className="num">
                    <Money value={order.fees_paid} signed={false} />
                  </td>
                  <td>
                    <StatusBadge status={order.status} />
                    {order.reject_reason ? (
                      <div
                        className="muted"
                        style={{ fontSize: "0.72rem", whiteSpace: "normal" }}
                      >
                        {order.reject_reason}
                      </div>
                    ) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty>No orders yet.</Empty>
      )}

      <Notice kind="info">
        There is deliberately no way to place a manual order here. Every order originates from a
        strategy signal that has passed the risk manager; a manual path would skip sizing,
        exposure checks and the kill switch.
      </Notice>

      <Disclaimer />
    </>
  );
}
