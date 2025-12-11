import { Layout, Section } from '../components';

export default function TermsPage() {
  return (
    <Layout title="Terms of Service" description="DharmaMind Terms of Service - Legal terms and conditions for using our platform">
      <Section background="white" padding="xl">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-4xl font-bold text-neutral-900 mb-4">Terms of Service</h1>
          <p className="text-lg text-neutral-600 mb-12">
            Last updated: {new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
          </p>

          <div className="prose prose-neutral max-w-none">
            <section className="mb-12">
              <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Welcome to DharmaMind</h2>
              <p className="text-neutral-600 mb-4">
                These Terms of Service (&quot;Terms&quot;) govern your access to and use of DharmaMind&apos;s AI spiritual guidance
                platform, including our website, mobile applications, and services (collectively, the &quot;Service&quot;).
              </p>
              <p className="text-neutral-600">
                By accessing or using our Service, you agree to be bound by these Terms. If you disagree with any
                part of these terms, you may not access the Service.
              </p>
            </section>

            <section className="mb-12">
              <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Acceptance of Terms</h2>
              <p className="text-neutral-600 mb-4">
                By creating an account or using our Service, you confirm that:
              </p>
              <ul className="list-disc list-inside text-neutral-600 space-y-2">
                <li>You are at least 18 years old or have parental consent</li>
                <li>You have the legal capacity to enter into these Terms</li>
                <li>You will use the Service in accordance with these Terms</li>
                <li>All information you provide is accurate and complete</li>
              </ul>
            </section>

            <section className="mb-12">
              <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Description of Service</h2>
              <p className="text-neutral-600 mb-4">
                DharmaMind provides an AI-powered spiritual guidance platform that offers:
              </p>
              <ul className="list-disc list-inside text-neutral-600 space-y-2">
                <li>Conversational AI for spiritual discussions and guidance</li>
                <li>Access to spiritual wisdom from various traditions</li>
                <li>Personalized meditation and mindfulness recommendations</li>
                <li>Community features for spiritual growth</li>
                <li>Educational resources about spiritual practices</li>
              </ul>
            </section>

            <section className="mb-12">
              <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Important Disclaimers</h2>
              <div className="bg-gold-50 border border-gold-200 rounded-lg p-4 mb-6">
                <p className="text-gold-700 font-medium">⚠️ DharmaMind is not a substitute for professional advice</p>
              </div>
              <p className="text-neutral-600 mb-4">
                Please understand that:
              </p>
              <ul className="list-disc list-inside text-neutral-600 space-y-2">
                <li><strong>Not Medical Advice:</strong> Our Service does not provide medical, psychological, or therapeutic advice</li>
                <li><strong>Not Professional Counseling:</strong> AI responses are not a replacement for licensed professionals</li>
                <li><strong>Spiritual Guidance Only:</strong> Information is provided for spiritual exploration and education</li>
                <li><strong>Seek Professional Help:</strong> For mental health concerns, please consult qualified professionals</li>
              </ul>
            </section>

            <section className="mb-12">
              <h2 className="text-2xl font-semibold text-neutral-900 mb-4">User Accounts</h2>
              <p className="text-neutral-600 mb-4">
                When you create an account with us, you must:
              </p>
              <ul className="list-disc list-inside text-neutral-600 space-y-2">
                <li>Provide accurate and complete information</li>
                <li>Maintain the security of your password</li>
                <li>Notify us immediately of any unauthorized access</li>
                <li>Accept responsibility for all activities under your account</li>
              </ul>
            </section>

            <section className="mb-12">
              <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Acceptable Use</h2>
              <p className="text-neutral-600 mb-4">
                You agree not to use our Service to:
              </p>
              <ul className="list-disc list-inside text-neutral-600 space-y-2">
                <li>Violate any laws or regulations</li>
                <li>Infringe on intellectual property rights</li>
                <li>Harass, abuse, or harm others</li>
                <li>Spread misinformation or harmful content</li>
                <li>Attempt to gain unauthorized access to our systems</li>
                <li>Use automated systems to access the Service without permission</li>
                <li>Interfere with the proper functioning of the Service</li>
              </ul>
            </section>

            <section className="mb-12">
              <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Subscription and Payments</h2>
              <p className="text-neutral-600 mb-4">
                For paid features:
              </p>
              <ul className="list-disc list-inside text-neutral-600 space-y-2">
                <li>Subscription fees are billed in advance on a recurring basis</li>
                <li>You may cancel your subscription at any time</li>
                <li>Refunds are provided according to our refund policy</li>
                <li>We may change pricing with reasonable notice</li>
                <li>Free trials may be offered at our discretion</li>
              </ul>
            </section>

            <section className="mb-12">
              <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Intellectual Property</h2>
              <p className="text-neutral-600 mb-4">
                The Service and its original content, features, and functionality are owned by DharmaMind and are
                protected by international copyright, trademark, and other intellectual property laws.
              </p>
              <ul className="list-disc list-inside text-neutral-600 space-y-2">
                <li>You retain rights to content you submit</li>
                <li>You grant us a license to use your content to provide the Service</li>
                <li>You may not copy, modify, or distribute our content without permission</li>
              </ul>
            </section>

            <section className="mb-12">
              <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Limitation of Liability</h2>
              <p className="text-neutral-600">
                To the maximum extent permitted by law, DharmaMind shall not be liable for any indirect, incidental,
                special, consequential, or punitive damages, or any loss of profits or revenues, whether incurred
                directly or indirectly, or any loss of data, use, goodwill, or other intangible losses resulting
                from your use of the Service.
              </p>
            </section>

            <section className="mb-12">
              <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Termination</h2>
              <p className="text-neutral-600 mb-4">
                We may terminate or suspend your account immediately, without prior notice or liability, for any reason,
                including without limitation if you breach these Terms.
              </p>
              <p className="text-neutral-600">
                Upon termination, your right to use the Service will immediately cease. You may request export of your
                data before account deletion.
              </p>
            </section>

            <section className="mb-12">
              <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Contact Us</h2>
              <p className="text-neutral-600 mb-4">
                If you have any questions about these Terms, please contact us:
              </p>
              <div className="bg-neutral-100 border border-neutral-200 rounded-lg p-4">
                <p className="text-neutral-700"><strong>Email:</strong> legal@dharmamind.com</p>
                <p className="text-neutral-700"><strong>Support:</strong> support@dharmamind.com</p>
              </div>
            </section>

            <section>
              <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Changes to Terms</h2>
              <p className="text-neutral-600">
                We reserve the right to modify or replace these Terms at any time. If a revision is material,
                we will provide at least 30 days&apos; notice prior to any new terms taking effect. What constitutes
                a material change will be determined at our sole discretion.
              </p>
            </section>
          </div>
        </div>
      </Section>
    </Layout>
  );
}
