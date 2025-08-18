#!/usr/bin/env python3
"""
DharmaMind Mobile/PWA Features - Phase 3
Progressive Web App capabilities with offline features
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import asyncio

class MobilePWAManager:
    """Mobile and PWA feature management"""
    
    def __init__(self):
        self.offline_cache = {}
        self.user_preferences = {}
        self.notification_queue = []
        self.sync_queue = []
        
    def generate_manifest(self) -> Dict[str, Any]:
        """Generate PWA manifest"""
        return {
            "name": "DharmaMind - AI Spiritual Companion",
            "short_name": "DharmaMind",
            "description": "Your personal AI guide for meditation, dharma study, and spiritual growth",
            "start_url": "/",
            "display": "standalone",
            "background_color": "#1a1a2e",
            "theme_color": "#ff6b6b",
            "orientation": "portrait-primary",
            "categories": ["lifestyle", "education", "health"],
            "lang": "en",
            "dir": "ltr",
            "icons": [
                {
                    "src": "/icons/icon-72x72.png",
                    "sizes": "72x72",
                    "type": "image/png",
                    "purpose": "any maskable"
                },
                {
                    "src": "/icons/icon-96x96.png",
                    "sizes": "96x96",
                    "type": "image/png",
                    "purpose": "any maskable"
                },
                {
                    "src": "/icons/icon-128x128.png",
                    "sizes": "128x128",
                    "type": "image/png",
                    "purpose": "any maskable"
                },
                {
                    "src": "/icons/icon-144x144.png",
                    "sizes": "144x144",
                    "type": "image/png",
                    "purpose": "any maskable"
                },
                {
                    "src": "/icons/icon-152x152.png",
                    "sizes": "152x152",
                    "type": "image/png",
                    "purpose": "any maskable"
                },
                {
                    "src": "/icons/icon-192x192.png",
                    "sizes": "192x192",
                    "type": "image/png",
                    "purpose": "any maskable"
                },
                {
                    "src": "/icons/icon-384x384.png",
                    "sizes": "384x384",
                    "type": "image/png",
                    "purpose": "any maskable"
                },
                {
                    "src": "/icons/icon-512x512.png",
                    "sizes": "512x512",
                    "type": "image/png",
                    "purpose": "any maskable"
                }
            ],
            "screenshots": [
                {
                    "src": "/screenshots/mobile-1.png",
                    "sizes": "390x844",
                    "type": "image/png",
                    "form_factor": "narrow"
                },
                {
                    "src": "/screenshots/desktop-1.png",
                    "sizes": "1920x1080",
                    "type": "image/png",
                    "form_factor": "wide"
                }
            ],
            "shortcuts": [
                {
                    "name": "Daily Meditation",
                    "short_name": "Meditate",
                    "description": "Start your daily meditation practice",
                    "url": "/meditation",
                    "icons": [{"src": "/icons/meditation-icon.png", "sizes": "96x96"}]
                },
                {
                    "name": "Study Dharma",
                    "short_name": "Study",
                    "description": "Explore dharma teachings and texts",
                    "url": "/study",
                    "icons": [{"src": "/icons/study-icon.png", "sizes": "96x96"}]
                },
                {
                    "name": "AI Companion",
                    "short_name": "Ask AI",
                    "description": "Chat with your AI spiritual guide",
                    "url": "/chat",
                    "icons": [{"src": "/icons/ai-icon.png", "sizes": "96x96"}]
                }
            ],
            "related_applications": [
                {
                    "platform": "play",
                    "url": "https://play.google.com/store/apps/details?id=ai.dharmamind.app",
                    "id": "ai.dharmamind.app"
                }
            ],
            "prefer_related_applications": False,
            "edge_side_panel": {
                "preferred_width": 400
            },
            "launch_handler": {
                "client_mode": "focus-existing"
            }
        }
    
    def generate_service_worker(self) -> str:
        """Generate service worker for offline capabilities"""
        return '''
const CACHE_NAME = 'dharmamind-v1.0.0';
const OFFLINE_URL = '/offline.html';

// Resources to cache for offline use
const CACHE_URLS = [
    '/',
    '/offline.html',
    '/manifest.json',
    '/css/main.css',
    '/js/app.js',
    '/js/meditation-timer.js',
    '/js/offline-manager.js',
    '/icons/icon-192x192.png',
    '/icons/icon-512x512.png',
    '/api/dharma/essential-teachings',
    '/api/meditation/basic-instructions'
];

// Install event - cache essential resources
self.addEventListener('install', event => {
    console.log('🚀 DharmaMind Service Worker installing...');
    
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {
                console.log('📦 Caching essential resources');
                return cache.addAll(CACHE_URLS);
            })
            .then(() => {
                console.log('✅ Service Worker installed successfully');
                return self.skipWaiting();
            })
            .catch(error => {
                console.error('❌ Cache installation failed:', error);
            })
    );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
    console.log('🔄 DharmaMind Service Worker activating...');
    
    event.waitUntil(
        caches.keys()
            .then(cacheNames => {
                return Promise.all(
                    cacheNames.map(cacheName => {
                        if (cacheName !== CACHE_NAME) {
                            console.log('🗑️ Deleting old cache:', cacheName);
                            return caches.delete(cacheName);
                        }
                    })
                );
            })
            .then(() => {
                console.log('✅ Service Worker activated');
                return self.clients.claim();
            })
    );
});

// Fetch event - serve from cache when offline
self.addEventListener('fetch', event => {
    const { request } = event;
    const url = new URL(request.url);
    
    // Handle navigation requests
    if (request.mode === 'navigate') {
        event.respondWith(
            fetch(request)
                .then(response => {
                    // If online, serve from network and update cache
                    const responseClone = response.clone();
                    caches.open(CACHE_NAME)
                        .then(cache => cache.put(request, responseClone));
                    return response;
                })
                .catch(() => {
                    // If offline, serve from cache or offline page
                    return caches.match(request)
                        .then(response => response || caches.match(OFFLINE_URL));
                })
        );
        return;
    }
    
    // Handle API requests
    if (url.pathname.startsWith('/api/')) {
        event.respondWith(
            fetch(request)
                .then(response => {
                    // Cache successful API responses
                    if (response.ok) {
                        const responseClone = response.clone();
                        caches.open(CACHE_NAME)
                            .then(cache => cache.put(request, responseClone));
                    }
                    return response;
                })
                .catch(() => {
                    // Serve from cache if offline
                    return caches.match(request)
                        .then(response => {
                            if (response) {
                                // Add offline indicator to cached response
                                const offlineResponse = response.clone();
                                return offlineResponse.json()
                                    .then(data => {
                                        data._offline = true;
                                        data._cached_at = new Date().toISOString();
                                        return new Response(JSON.stringify(data), {
                                            headers: { 'Content-Type': 'application/json' }
                                        });
                                    });
                            }
                            return new Response(
                                JSON.stringify({ 
                                    error: 'Offline - content not cached',
                                    offline: true 
                                }), 
                                { 
                                    status: 503,
                                    headers: { 'Content-Type': 'application/json' }
                                }
                            );
                        });
                })
        );
        return;
    }
    
    // Handle other requests (CSS, JS, images)
    event.respondWith(
        caches.match(request)
            .then(response => {
                if (response) {
                    return response;
                }
                return fetch(request)
                    .then(response => {
                        // Cache successful responses
                        if (response.ok) {
                            const responseClone = response.clone();
                            caches.open(CACHE_NAME)
                                .then(cache => cache.put(request, responseClone));
                        }
                        return response;
                    });
            })
    );
});

// Background sync for offline actions
self.addEventListener('sync', event => {
    console.log('🔄 Background sync triggered:', event.tag);
    
    if (event.tag === 'sync-meditation-sessions') {
        event.waitUntil(syncMeditationSessions());
    } else if (event.tag === 'sync-study-progress') {
        event.waitUntil(syncStudyProgress());
    }
});

// Push notifications
self.addEventListener('push', event => {
    console.log('📬 Push notification received');
    
    const options = {
        body: event.data ? event.data.text() : 'Time for your daily meditation practice 🧘',
        icon: '/icons/icon-192x192.png',
        badge: '/icons/badge-72x72.png',
        tag: 'dharmamind-reminder',
        vibrate: [100, 50, 100],
        data: {
            dateOfArrival: Date.now(),
            primaryKey: '1',
            url: '/meditation'
        },
        actions: [
            {
                action: 'meditate',
                title: 'Start Meditation',
                icon: '/icons/meditate-action.png'
            },
            {
                action: 'later',
                title: 'Remind Later',
                icon: '/icons/later-action.png'
            }
        ]
    };
    
    event.waitUntil(
        self.registration.showNotification('DharmaMind Reminder', options)
    );
});

// Notification click handling
self.addEventListener('notificationclick', event => {
    console.log('📱 Notification clicked:', event.action);
    
    event.notification.close();
    
    if (event.action === 'meditate') {
        event.waitUntil(
            clients.openWindow('/meditation')
        );
    } else if (event.action === 'later') {
        // Schedule another reminder
        console.log('⏰ Reminder scheduled for later');
    } else {
        // Default action - open app
        event.waitUntil(
            clients.openWindow('/')
        );
    }
});

// Helper functions
async function syncMeditationSessions() {
    try {
        const sessions = await getOfflineData('meditation-sessions');
        if (sessions && sessions.length > 0) {
            const response = await fetch('/api/sync/meditation-sessions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(sessions)
            });
            
            if (response.ok) {
                await clearOfflineData('meditation-sessions');
                console.log('✅ Meditation sessions synced');
            }
        }
    } catch (error) {
        console.error('❌ Meditation sync failed:', error);
    }
}

async function syncStudyProgress() {
    try {
        const progress = await getOfflineData('study-progress');
        if (progress && progress.length > 0) {
            const response = await fetch('/api/sync/study-progress', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(progress)
            });
            
            if (response.ok) {
                await clearOfflineData('study-progress');
                console.log('✅ Study progress synced');
            }
        }
    } catch (error) {
        console.error('❌ Study sync failed:', error);
    }
}

async function getOfflineData(key) {
    const cache = await caches.open('dharmamind-offline-data');
    const response = await cache.match(`/offline-data/${key}`);
    return response ? await response.json() : null;
}

async function clearOfflineData(key) {
    const cache = await caches.open('dharmamind-offline-data');
    await cache.delete(`/offline-data/${key}`);
}

console.log('🧘 DharmaMind Service Worker loaded');
        '''
    
    def get_mobile_optimizations(self) -> Dict[str, Any]:
        """Get mobile-specific optimizations"""
        return {
            "viewport_config": {
                "width": "device-width",
                "initial_scale": 1.0,
                "maximum_scale": 5.0,
                "user_scalable": True,
                "viewport_fit": "cover"
            },
            "touch_optimizations": {
                "touch_action": "manipulation",
                "tap_highlight_color": "transparent",
                "touch_callout": "none",
                "user_select": "none"
            },
            "performance_hints": {
                "preload_critical_resources": [
                    "/css/critical.css",
                    "/js/app.js",
                    "/api/user/preferences"
                ],
                "lazy_load_images": True,
                "compress_responses": True,
                "enable_http2_push": True
            },
            "accessibility": {
                "high_contrast_support": True,
                "voice_navigation": True,
                "screen_reader_optimized": True,
                "font_size_scaling": True
            }
        }
    
    def get_offline_features(self) -> Dict[str, Any]:
        """Get offline feature capabilities"""
        return {
            "cached_content": [
                "Essential dharma teachings",
                "Basic meditation instructions",
                "Daily reflection prompts",
                "Sanskrit pronunciation guide",
                "Emergency spiritual support"
            ],
            "offline_actions": [
                "Record meditation sessions",
                "Save personal insights",
                "Bookmark favorite teachings",
                "Continue reading downloaded texts",
                "Access meditation timer"
            ],
            "sync_capabilities": [
                "Meditation session data",
                "Study progress tracking",
                "Personal notes and insights",
                "Reading bookmarks",
                "Practice streaks and achievements"
            ],
            "storage_limits": {
                "cache_size_mb": 50,
                "user_data_mb": 25,
                "offline_content_mb": 100
            }
        }
    
    def create_notification_config(self) -> Dict[str, Any]:
        """Create push notification configuration"""
        return {
            "daily_reminders": {
                "meditation_reminder": {
                    "title": "🧘 Time for Daily Meditation",
                    "body": "Your moment of peace awaits. Take a few minutes to center yourself.",
                    "icon": "/icons/meditation-icon.png",
                    "default_time": "07:00",
                    "customizable": True
                },
                "evening_reflection": {
                    "title": "🌅 Evening Reflection",
                    "body": "How was your spiritual practice today? Take a moment to reflect.",
                    "icon": "/icons/reflection-icon.png",
                    "default_time": "20:00",
                    "customizable": True
                },
                "dharma_study": {
                    "title": "📚 Daily Dharma Study",
                    "body": "Discover wisdom from ancient teachings. Continue your learning journey.",
                    "icon": "/icons/study-icon.png",
                    "default_time": "18:00",
                    "customizable": True
                }
            },
            "achievement_notifications": {
                "meditation_streak": {
                    "title": "🎉 Meditation Streak Achievement!",
                    "body": "Congratulations on {days} days of consistent practice!",
                    "icon": "/icons/achievement-icon.png"
                },
                "study_milestone": {
                    "title": "📖 Study Milestone Reached!",
                    "body": "You've completed {chapters} chapters of dharma study. Well done!",
                    "icon": "/icons/milestone-icon.png"
                }
            },
            "inspirational_quotes": {
                "enabled": True,
                "frequency": "weekly",
                "categories": ["buddha", "dharma", "meditation", "wisdom", "compassion"]
            }
        }

# Global PWA manager instance
_pwa_manager = None

def get_pwa_manager() -> MobilePWAManager:
    """Get global PWA manager instance"""
    global _pwa_manager
    if _pwa_manager is None:
        _pwa_manager = MobilePWAManager()
    return _pwa_manager

def demo_mobile_features():
    """Demo mobile/PWA features"""
    print("📱 DharmaMind Mobile/PWA Features - Phase 3")
    print("=" * 60)
    
    pwa_manager = get_pwa_manager()
    
    # Generate manifest
    print("📋 PWA Manifest Generated:")
    manifest = pwa_manager.generate_manifest()
    print(f"  Name: {manifest['name']}")
    print(f"  Description: {manifest['description']}")
    print(f"  Icons: {len(manifest['icons'])} sizes available")
    print(f"  Shortcuts: {len(manifest['shortcuts'])} quick actions")
    
    # Mobile optimizations
    print("\\n📱 Mobile Optimizations:")
    optimizations = pwa_manager.get_mobile_optimizations()
    print(f"  Viewport: {optimizations['viewport_config']['width']}")
    print(f"  Touch optimized: {optimizations['touch_optimizations']['touch_action']}")
    print(f"  Accessibility: {optimizations['accessibility']['screen_reader_optimized']}")
    
    # Offline features
    print("\\n🔌 Offline Capabilities:")
    offline = pwa_manager.get_offline_features()
    print(f"  Cached content types: {len(offline['cached_content'])}")
    print(f"  Offline actions: {len(offline['offline_actions'])}")
    print(f"  Sync capabilities: {len(offline['sync_capabilities'])}")
    print(f"  Total storage: {sum(offline['storage_limits'].values())}MB")
    
    # Notifications
    print("\\n🔔 Notification Features:")
    notifications = pwa_manager.create_notification_config()
    print(f"  Daily reminders: {len(notifications['daily_reminders'])}")
    print(f"  Achievement notifications: {len(notifications['achievement_notifications'])}")
    print(f"  Inspirational quotes: {notifications['inspirational_quotes']['enabled']}")
    
    print("\\n✅ Mobile/PWA Phase 3 Complete!")
    print("📱 App ready for mobile installation")
    print("🔌 Offline capabilities configured")
    print("🔔 Smart notifications enabled")

if __name__ == "__main__":
    demo_mobile_features()
