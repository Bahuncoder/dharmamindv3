import React, { useState } from 'react';
import { useNotifications } from '../contexts/NotificationContext';

const NotificationCenter: React.FC = () => {
    const { notifications, markAsRead, removeNotification, unreadCount } = useNotifications();
    const [isOpen, setIsOpen] = useState(false);

    const getIcon = (type: string) => {
        const icons = {
            info: '📘',
            success: '✅',
            warning: '⚠️',
            error: '❌',
        };
        return icons[type as keyof typeof icons] || '📢';
    };

    const getTypeColor = (type: string) => {
        const colors = {
            info: 'border-l-info bg-info-light',
            success: 'border-l-success bg-success-light',
            warning: 'border-l-warning bg-warning-light',
            error: 'border-l-error bg-error-light',
        };
        return colors[type as keyof typeof colors] || 'border-l-neutral-400 bg-neutral-50';
    };

    const formatTime = (timestamp: Date) => {
        const now = new Date();
        const diff = now.getTime() - timestamp.getTime();
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);

        if (days > 0) return `${days}d ago`;
        if (hours > 0) return `${hours}h ago`;
        if (minutes > 0) return `${minutes}m ago`;
        return 'Just now';
    };

    return (
        <div className="relative">
            {/* Notification Bell */}
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="relative p-2 text-secondary hover:text-primary transition-colors duration-200"
                aria-label="Notifications"
            >
                <span className="text-xl">🔔</span>
                {unreadCount > 0 && (
                    <span className="absolute -top-1 -right-1 bg-error text-white text-xs rounded-full h-5 w-5 flex items-center justify-center font-semibold">
                        {unreadCount > 9 ? '9+' : unreadCount}
                    </span>
                )}
            </button>

            {/* Notification Panel */}
            {isOpen && (
                <>
                    {/* Overlay */}
                    <div
                        className="fixed inset-0 z-40"
                        onClick={() => setIsOpen(false)}
                    />

                    {/* Notification Dropdown */}
                    <div className="absolute right-0 top-full mt-2 w-80 bg-primary shadow-xl rounded-lg border border-border-light z-50 max-h-96 overflow-hidden">
                        <div className="p-4 border-b border-border-light">
                            <div className="flex items-center justify-between">
                                <h3 className="text-lg font-bold text-primary">Notifications</h3>
                                {unreadCount > 0 && (
                                    <span className="text-sm text-secondary font-semibold">
                                        {unreadCount} unread
                                    </span>
                                )}
                            </div>
                        </div>

                        <div className="max-h-80 overflow-y-auto">
                            {notifications.length === 0 ? (
                                <div className="p-8 text-center">
                                    <span className="text-4xl mb-3 block">🔕</span>
                                    <p className="text-secondary font-semibold">No notifications yet</p>
                                </div>
                            ) : (
                                notifications.map((notification) => (
                                    <div
                                        key={notification.id}
                                        className={`p-4 border-l-4 ${getTypeColor(notification.type)} ${!notification.read ? 'bg-opacity-50' : 'bg-opacity-20'
                                            } hover:bg-opacity-60 transition-all duration-200 cursor-pointer`}
                                        onClick={() => markAsRead(notification.id)}
                                    >
                                        <div className="flex items-start gap-3">
                                            <span className="text-lg flex-shrink-0 mt-0.5">
                                                {getIcon(notification.type)}
                                            </span>
                                            <div className="flex-1 min-w-0">
                                                <div className="flex items-center justify-between">
                                                    <h4 className={`text-sm font-bold ${!notification.read ? 'text-primary' : 'text-secondary'}`}>
                                                        {notification.title}
                                                    </h4>
                                                    <button
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            removeNotification(notification.id);
                                                        }}
                                                        className="text-muted hover:text-secondary transition-colors text-xs ml-2"
                                                    >
                                                        ✕
                                                    </button>
                                                </div>
                                                <p className="text-sm text-secondary mt-1 line-clamp-2">
                                                    {notification.message}
                                                </p>
                                                <div className="flex items-center justify-between mt-2">
                                                    <span className="text-xs text-muted font-medium">
                                                        {formatTime(notification.timestamp)}
                                                    </span>
                                                    {!notification.read && (
                                                        <span className="w-2 h-2 bg-primary rounded-full"></span>
                                                    )}
                                                </div>
                                                {notification.action && (
                                                    <button
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            notification.action!.onClick();
                                                            markAsRead(notification.id);
                                                        }}
                                                        className="mt-2 text-xs bg-primary text-white px-3 py-1 rounded font-semibold hover:opacity-90 transition-opacity"
                                                    >
                                                        {notification.action.label}
                                                    </button>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>

                        {notifications.length > 0 && (
                            <div className="p-3 border-t border-border-light bg-bg-tertiary">
                                <button
                                    onClick={() => {
                                        notifications.forEach(n => markAsRead(n.id));
                                        setIsOpen(false);
                                    }}
                                    className="w-full text-sm font-semibold text-secondary hover:text-primary transition-colors"
                                >
                                    Mark all as read
                                </button>
                            </div>
                        )}
                    </div>
                </>
            )}
        </div>
    );
};

export default NotificationCenter;
