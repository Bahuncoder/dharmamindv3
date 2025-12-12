/**
 * DharmaMind Billing Page
 * =======================
 * View billing history, invoices, and manage payment methods
 */

import React, { useState } from 'react';
import Layout from '../components/layout/Layout';
import Link from 'next/link';

interface Invoice {
    id: string;
    date: string;
    amount: number;
    status: 'paid' | 'pending' | 'failed';
    description: string;
}

interface PaymentMethod {
    id: string;
    type: 'card';
    brand: string;
    last4: string;
    expMonth: number;
    expYear: number;
    isDefault: boolean;
}

// Mock data - would come from API
const mockInvoices: Invoice[] = [
    { id: 'inv_001', date: '2024-12-01', amount: 19, status: 'paid', description: 'Pro Plan - Monthly' },
    { id: 'inv_002', date: '2024-11-01', amount: 19, status: 'paid', description: 'Pro Plan - Monthly' },
    { id: 'inv_003', date: '2024-10-01', amount: 19, status: 'paid', description: 'Pro Plan - Monthly' },
];

const mockPaymentMethods: PaymentMethod[] = [
    { id: 'pm_001', type: 'card', brand: 'Visa', last4: '4242', expMonth: 12, expYear: 2025, isDefault: true },
];

export default function BillingPage() {
    const [invoices] = useState<Invoice[]>(mockInvoices);
    const [paymentMethods] = useState<PaymentMethod[]>(mockPaymentMethods);
    const [showAddCard, setShowAddCard] = useState(false);

    const currentPlan = {
        name: 'Pro',
        price: 19,
        billing: 'monthly',
        nextBilling: '2025-01-01',
    };

    const getStatusBadge = (status: Invoice['status']) => {
        const styles = {
            paid: 'bg-green-100 text-green-700',
            pending: 'bg-yellow-100 text-yellow-700',
            failed: 'bg-red-100 text-red-700',
        };
        return (
            <span className={`px-2 py-1 text-xs font-medium rounded-full ${styles[status]}`}>
                {status.charAt(0).toUpperCase() + status.slice(1)}
            </span>
        );
    };

    return (
        <Layout
            title="Billing"
            description="Manage your billing and payment methods"
        >
            <div className="min-h-screen bg-neutral-100 py-12 px-6">
                <div className="max-w-4xl mx-auto">
                    {/* Header */}
                    <div className="mb-8">
                        <h1 className="text-3xl font-semibold text-neutral-900 mb-2">Billing</h1>
                        <p className="text-neutral-600">Manage your subscription, payment methods, and invoices.</p>
                    </div>

                    {/* Current Plan */}
                    <div className="bg-white p-6 rounded-2xl border border-neutral-200 mb-8">
                        <div className="flex items-center justify-between">
                            <div>
                                <h2 className="text-lg font-semibold text-neutral-900 mb-1">Current Plan</h2>
                                <p className="text-neutral-600">
                                    <span className="font-medium text-gold-600">{currentPlan.name}</span>
                                    {' • '}${currentPlan.price}/{currentPlan.billing}
                                </p>
                                <p className="text-sm text-neutral-500 mt-1">
                                    Next billing: {new Date(currentPlan.nextBilling).toLocaleDateString()}
                                </p>
                            </div>
                            <div className="flex gap-3">
                                <Link
                                    href="/pricing"
                                    className="px-4 py-2 bg-neutral-100 text-neutral-700 font-medium rounded-lg hover:bg-neutral-200 transition-colors"
                                >
                                    Change Plan
                                </Link>
                                <button className="px-4 py-2 text-red-600 font-medium hover:bg-red-50 rounded-lg transition-colors">
                                    Cancel
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Payment Methods */}
                    <div className="bg-white p-6 rounded-2xl border border-neutral-200 mb-8">
                        <div className="flex items-center justify-between mb-6">
                            <h2 className="text-lg font-semibold text-neutral-900">Payment Methods</h2>
                            <button
                                onClick={() => setShowAddCard(true)}
                                className="px-4 py-2 bg-gold-600 text-white font-medium rounded-lg hover:bg-gold-700 transition-colors"
                            >
                                Add Card
                            </button>
                        </div>

                        <div className="space-y-4">
                            {paymentMethods.map((method) => (
                                <div
                                    key={method.id}
                                    className="flex items-center justify-between p-4 border border-neutral-200 rounded-xl"
                                >
                                    <div className="flex items-center gap-4">
                                        {/* Card Icon */}
                                        <div className="w-12 h-8 bg-neutral-100 rounded flex items-center justify-center">
                                            <span className="text-xs font-medium text-neutral-600">{method.brand}</span>
                                        </div>
                                        <div>
                                            <p className="font-medium text-neutral-900">
                                                •••• •••• •••• {method.last4}
                                            </p>
                                            <p className="text-sm text-neutral-500">
                                                Expires {method.expMonth}/{method.expYear}
                                            </p>
                                        </div>
                                        {method.isDefault && (
                                            <span className="px-2 py-1 text-xs font-medium bg-gold-100 text-gold-700 rounded-full">
                                                Default
                                            </span>
                                        )}
                                    </div>
                                    <div className="flex gap-2">
                                        {!method.isDefault && (
                                            <button className="text-sm text-neutral-600 hover:text-neutral-900">
                                                Set Default
                                            </button>
                                        )}
                                        <button className="text-sm text-red-600 hover:text-red-700">
                                            Remove
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Add Card Modal */}
                        {showAddCard && (
                            <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                                <div className="bg-white p-8 rounded-2xl max-w-md w-full mx-4">
                                    <h3 className="text-xl font-semibold text-neutral-900 mb-6">Add Payment Method</h3>
                                    <form className="space-y-4">
                                        <div>
                                            <label className="block text-sm font-medium text-neutral-700 mb-2">
                                                Card Number
                                            </label>
                                            <input
                                                type="text"
                                                className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-600 focus:border-transparent"
                                                placeholder="1234 5678 9012 3456"
                                            />
                                        </div>
                                        <div className="grid grid-cols-2 gap-4">
                                            <div>
                                                <label className="block text-sm font-medium text-neutral-700 mb-2">
                                                    Expiry
                                                </label>
                                                <input
                                                    type="text"
                                                    className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-600 focus:border-transparent"
                                                    placeholder="MM/YY"
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-neutral-700 mb-2">
                                                    CVC
                                                </label>
                                                <input
                                                    type="text"
                                                    className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-gold-600 focus:border-transparent"
                                                    placeholder="123"
                                                />
                                            </div>
                                        </div>
                                        <div className="flex gap-3 pt-4">
                                            <button
                                                type="button"
                                                onClick={() => setShowAddCard(false)}
                                                className="flex-1 px-4 py-3 bg-neutral-100 text-neutral-700 font-medium rounded-lg hover:bg-neutral-200 transition-colors"
                                            >
                                                Cancel
                                            </button>
                                            <button
                                                type="submit"
                                                className="flex-1 px-4 py-3 bg-gold-600 text-white font-medium rounded-lg hover:bg-gold-700 transition-colors"
                                            >
                                                Add Card
                                            </button>
                                        </div>
                                    </form>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Billing History */}
                    <div className="bg-white p-6 rounded-2xl border border-neutral-200">
                        <h2 className="text-lg font-semibold text-neutral-900 mb-6">Billing History</h2>

                        <div className="overflow-x-auto">
                            <table className="w-full">
                                <thead>
                                    <tr className="border-b border-neutral-200">
                                        <th className="text-left py-3 px-4 text-sm font-medium text-neutral-500">Date</th>
                                        <th className="text-left py-3 px-4 text-sm font-medium text-neutral-500">Description</th>
                                        <th className="text-left py-3 px-4 text-sm font-medium text-neutral-500">Amount</th>
                                        <th className="text-left py-3 px-4 text-sm font-medium text-neutral-500">Status</th>
                                        <th className="text-right py-3 px-4 text-sm font-medium text-neutral-500">Invoice</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {invoices.map((invoice) => (
                                        <tr key={invoice.id} className="border-b border-neutral-100">
                                            <td className="py-4 px-4 text-neutral-900">
                                                {new Date(invoice.date).toLocaleDateString()}
                                            </td>
                                            <td className="py-4 px-4 text-neutral-600">{invoice.description}</td>
                                            <td className="py-4 px-4 text-neutral-900 font-medium">${invoice.amount}</td>
                                            <td className="py-4 px-4">{getStatusBadge(invoice.status)}</td>
                                            <td className="py-4 px-4 text-right">
                                                <button className="text-gold-600 hover:text-gold-700 text-sm font-medium">
                                                    Download
                                                </button>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        {invoices.length === 0 && (
                            <div className="text-center py-12">
                                <p className="text-neutral-500">No invoices yet</p>
                            </div>
                        )}
                    </div>

                    {/* Help */}
                    <div className="mt-8 text-center">
                        <p className="text-neutral-600">
                            Need help with billing?{' '}
                            <Link href="/contact" className="text-gold-600 hover:underline">
                                Contact support
                            </Link>
                        </p>
                    </div>
                </div>
            </div>
        </Layout>
    );
}
