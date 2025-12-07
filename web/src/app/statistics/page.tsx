import { getStatsFolders } from '@/app/actions/stats';
import StatsViewer from '@/components/StatsViewer';

export const dynamic = 'force-dynamic';

export default async function StatisticsPage() {
    const folders = await getStatsFolders();

    return (
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
            <div className="px-4 py-6 sm:px-0">
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-6">
                    Training Statistics
                </h1>
                <StatsViewer folders={folders} />
            </div>
        </div>
    );
}
