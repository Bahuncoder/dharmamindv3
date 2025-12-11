import Link from 'next/link';
import { JobOpening } from '../../config/site.config';

interface JobCardProps {
    job: JobOpening;
}

export default function JobCard({ job }: JobCardProps) {
    return (
        <Link href={`/careers/${job.id}`} className="block group">
            <div className="p-6 rounded-2xl border border-neutral-200 bg-neutral-100 hover:border-neutral-300 hover:shadow-md transition-all">
                <div className="flex justify-between items-start mb-3">
                    <div>
                        <h3 className="text-lg font-semibold text-neutral-900 group-hover:text-neutral-700 transition-colors">
                            {job.title}
                        </h3>
                        <p className="text-sm text-neutral-600">{job.department}</p>
                    </div>
                    <span className="text-2xl opacity-60 group-hover:opacity-100 transition-opacity">→</span>
                </div>

                <p className="text-neutral-600 text-sm mb-4 line-clamp-2">
                    {job.description}
                </p>

                <div className="flex flex-wrap gap-2">
                    <span className="px-3 py-1 bg-neutral-100 text-neutral-600 text-xs rounded-full">
                        {job.location}
                    </span>
                    <span className="px-3 py-1 bg-neutral-100 text-neutral-600 text-xs rounded-full">
                        {job.type}
                    </span>
                </div>
            </div>
        </Link>
    );
}
