import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface NotificationContextType {
    notifications: Notification[];
    addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => void;
    removeNotification: (id: string) => void;
    markAsRead: (id: string) => void;
    unreadCount: number;
}

interface Notification {
    id: string;
    type: 'info' | 'success' | 'warning' | 'error';
    title: string;
    message: string;
    timestamp: Date;
    read: boolean;
    action?: {
        label: string;
        onClick: () => void;
    };
}

const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

export const useNotifications = (): NotificationContextType => {
    const context = useContext(NotificationContext);
    if (!context) {
        throw new Error('useNotifications must be used within a NotificationProvider');
    }
    return context;
};

interface NotificationProviderProps {
    children: ReactNode;
}

export const NotificationProvider: React.FC<NotificationProviderProps> = ({ children }) => {
    const [notifications, setNotifications] = useState<Notification[]>([
        {
            id: '1',
            type: 'info',
            title: 'Welcome to DharmaMind Community!',
            message: 'Explore discussions, connect with like-minded souls, and grow together.',
            timestamp: new Date(),
            read: false,
        },
        {
            id: '2',
            type: 'success',
            title: 'New Discussion Started',
            message: 'Someone started a discussion on "Mindful Living in the Digital Age"',
            timestamp: new Date(Date.now() - 30 * 60 * 1000),
            read: false,
        }
    ]);

    const addNotification = (notificationData: Omit<Notification, 'id' | 'timestamp'>) => {
        const newNotification: Notification = {
            ...notificationData,
            id: Date.now().toString(),
            timestamp: new Date(),
        };
        setNotifications(prev => [newNotification, ...prev]);
    };

    const removeNotification = (id: string) => {
        setNotifications(prev => prev.filter(notification => notification.id !== id));
    };

    const markAsRead = (id: string) => {
        setNotifications(prev =>
            prev.map(notification =>
                notification.id === id ? { ...notification, read: true } : notification
            )
        );
    };

    const unreadCount = notifications.filter(n => !n.read).length;

    return (
        <NotificationContext.Provider
            value={{
                notifications,
                addNotification,
                removeNotification,
                markAsRead,
                unreadCount
            }}
        >
            {children}
        </NotificationContext.Provider>
    );
};

export default NotificationContext;
